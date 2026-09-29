"""Authentication recovery with fake accounts and no requests to Apple."""
import ast
import io
import json
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, call, patch

from pyicloud import PyiCloudService
from pyicloud.exceptions import PyiCloud2FARequiredException, PyiCloudFailedLoginException
from requests import Response
from requests.exceptions import ConnectionError

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from icloud_service import ICloudService, ICloudAuthenticationError
from calendar_api import ICloudCalendar


class AuthTests(unittest.TestCase):
    def setUp(self):
        network = patch('requests.sessions.Session.request', side_effect=AssertionError('Unexpected network access'))
        network.start()
        self.addCleanup(network.stop)
        self.api = Mock(spec=PyiCloudService)
        self.api.requires_2fa = True
        self.api.requires_2sa = True
        self.api.is_trusted_session = False
        self.api.security_key_names = None
        self.api.two_factor_delivery_method = 'sms'
        self.api.two_factor_delivery_notice = None
        self.api.request_2fa_code.return_value = True
        self.api._trusted_device_bridge_state = None
        self.api.trust_session.side_effect = self.accept
        self.service = ICloudService.__new__(ICloudService)
        self.service.api = self.api

    def accept(self, *args):
        self.api.requires_2fa = self.api.requires_2sa = False
        self.api.is_trusted_session = True
        return True

    def run_auth(self, answers):
        output = io.StringIO()
        with redirect_stdout(output), patch('builtins.input', side_effect=answers) as prompt:
            result = self.service.handle_authentication()
        return result, output.getvalue(), prompt

    def test_sms_2fa_exception_from_real_validator_can_retry(self):
        response = Response()
        response.status_code = 421
        # Exercise the exact SDK path from the Pi traceback: this exception is
        # not swallowed by pyicloud's APIResponseException handler.
        error = PyiCloud2FARequiredException('fixture@example.invalid', response)
        self.api._validate_sms_code.side_effect = [error, None]
        self.api.validate_2fa_code.side_effect = lambda code: PyiCloudService.validate_2fa_code(self.api, code)
        result, output, prompts = self.run_auth(['123456', '654321'])
        self.assertTrue(result)
        self.api.request_2fa_code.assert_called_once_with()
        self.assertEqual(self.api._validate_sms_code.call_args_list, [call('123456'), call('654321')])
        self.assertTrue(all('SMS code' in args.args[0] for args in prompts.call_args_list))
        self.assertIn('PyiCloud2FARequiredException', output)
        self.assertNotIn('fixture@example.invalid', output)
        self.assertNotIn('123456', output)

    def test_false_result_retries_without_resending(self):
        results = iter([False, True])
        self.api.validate_2fa_code.side_effect = lambda code: self.accept() if next(results) else False
        result, _, _ = self.run_auth(['123456', ' 000 123 '])
        self.assertTrue(result)
        self.api.request_2fa_code.assert_called_once()
        self.assertEqual(self.api.validate_2fa_code.call_args_list, [call('123456'), call('000123')])

    def test_invalid_input_does_not_use_attempts_or_send_more_codes(self):
        self.api.validate_2fa_code.side_effect = self.accept
        result, _, _ = self.run_auth(['nope', '123', '1234567', '123456'])
        self.assertTrue(result)
        self.api.validate_2fa_code.assert_called_once_with('123456')
        self.api.request_2fa_code.assert_called_once()

    def test_resend_is_explicit(self):
        self.api.validate_2fa_code.side_effect = self.accept
        result, _, _ = self.run_auth(['r', '123456'])
        self.assertTrue(result)
        self.assertEqual(self.api.request_2fa_code.call_count, 2)

    def test_trusted_device_attempt_requires_a_fresh_explicit_challenge(self):
        self.api.two_factor_delivery_method = 'trusted_device'
        results = iter([False, True])
        self.api.validate_2fa_code.side_effect = lambda code: self.accept() if next(results) else False
        result, output, _ = self.run_auth(['123456', '111111', 'r', '654321'])
        self.assertTrue(result)
        self.assertIn('previous trusted-device challenge ended', output)
        self.assertEqual(self.api.request_2fa_code.call_count, 2)
        self.assertEqual(self.api.validate_2fa_code.call_args_list, [call('123456'), call('654321')])

    def test_three_failed_attempts_return_without_exiting(self):
        self.api.validate_2fa_code.return_value = False
        result, output, _ = self.run_auth(['111111', '222222', '333333'])
        self.assertFalse(result)
        self.assertIn('Verification attempt limit reached', output)
        self.assertEqual(self.api.validate_2fa_code.call_count, 3)
        self.api.request_2fa_code.assert_called_once()

    def test_resends_are_bounded(self):
        result, output, _ = self.run_auth(['r', 'r', 'r'])
        self.assertFalse(result)
        self.assertIn('Code request limit reached', output)
        self.assertEqual(self.api.request_2fa_code.call_count, 3)
        self.api.validate_2fa_code.assert_not_called()

    def test_skip_and_noninteractive_input_leave_calendar_unverified(self):
        for answer in ('', 's', EOFError()):
            with self.subTest(answer=answer):
                result, _, _ = self.run_auth([answer])
                self.assertFalse(result)
        self.api.validate_2fa_code.assert_not_called()
        self.api.trust_session.assert_not_called()

    def test_delivery_failure_does_not_ask_for_an_unavailable_code(self):
        for result, error in [(False, None), (True, ConnectionError('fixture'))]:
            with self.subTest(error=error):
                self.api.request_2fa_code.return_value = result
                self.api.request_2fa_code.side_effect = error
                authenticated, _, prompts = self.run_auth([])
                self.assertFalse(authenticated)
                prompts.assert_not_called()

    def test_network_failure_during_validation_can_retry(self):
        results = iter([ConnectionError('fixture'), True])
        def validate(code):
            result = next(results)
            if isinstance(result, Exception):
                raise result
            return self.accept()
        self.api.validate_2fa_code.side_effect = validate
        result, _, _ = self.run_auth(['123456', '123456'])
        self.assertTrue(result)
        self.api.request_2fa_code.assert_called_once()

    def test_trusted_session_needs_no_code_delivery(self):
        self.accept()
        result, _, prompt = self.run_auth([])
        self.assertTrue(result)
        self.api.request_2fa_code.assert_not_called()
        prompt.assert_not_called()

    def test_trust_failure_does_not_grant_calendar_access(self):
        self.api.validate_2fa_code.return_value = True
        self.api.trust_session.side_effect = None
        self.api.trust_session.return_value = False
        result, output, _ = self.run_auth(['123456'])
        self.assertFalse(result)
        self.assertIn('trust did not complete', output)

    def test_missing_security_key_does_not_prompt_for_sms(self):
        self.api.security_key_names = ['fixture security key']
        self.api.fido2_devices = []
        result, _, prompt = self.run_auth([])
        self.assertFalse(result)
        prompt.assert_not_called()
        self.api.request_2fa_code.assert_not_called()

    def test_two_step_codes_can_retry_without_resending(self):
        self.api.requires_2fa = False
        self.api.trusted_devices = [{'deviceName': 'fixture phone'}]
        self.api.send_verification_code.return_value = True
        results = iter([False, True])
        self.api.validate_verification_code.side_effect = lambda device, code: self.accept() if next(results) else False
        with patch('icloud_service.click.prompt', return_value=0):
            result, _, _ = self.run_auth(['1234', '4321'])
        self.assertTrue(result)
        self.api.send_verification_code.assert_called_once()
        self.assertEqual(self.api.validate_verification_code.call_count, 2)

    def test_icloud_terms_acceptance_is_preserved(self):
        self.accept()
        with patch('icloud_service.load_dotenv'), patch('icloud_service.PyiCloudService', return_value=self.api) as create:
            service = ICloudService('fixture@example.invalid', 'fixture-password')
        create.assert_called_once_with('fixture@example.invalid', 'fixture-password', accept_terms=True)
        self.assertIs(service.api, self.api)

    def test_rejected_codes_allow_the_actual_calendar_constructor_to_finish(self):
        self.api.validate_2fa_code.return_value = False
        environment = {'APPLE_ID': 'fixture@example.invalid', 'ICLOUD_PWD': 'fixture-password'}
        with patch.dict('os.environ', environment), patch('icloud_service.load_dotenv'), \
                patch('icloud_service.PyiCloudService', return_value=self.api), \
                patch('builtins.input', side_effect=['111111', '222222', '333333']), \
                redirect_stdout(io.StringIO()) as output:
            calendar = ICloudCalendar()
        self.assertIsNone(calendar.service)
        self.assertIn('HAL will continue without calendar access', output.getvalue())
        self.assertIn('No calendar data', calendar.dispatch('calendar_next_event')['error'])
        self.api.request_2fa_code.assert_called_once()


class CalendarAvailabilityTests(unittest.TestCase):
    def test_skipped_verification_keeps_backend_available_to_report_errors(self):
        with patch('calendar_api.ICloudService', side_effect=ICloudAuthenticationError('Verification skipped.')):
            with redirect_stdout(io.StringIO()) as output:
                calendar = ICloudCalendar()
        self.assertIn('HAL will continue without calendar access', output.getvalue())
        result = calendar.dispatch('calendar_next_event')
        self.assertIn('unavailable', result['error'])
        self.assertIn('No calendar data', result['error'])

    def test_login_and_network_failures_do_not_abort_startup(self):
        for error in (PyiCloudFailedLoginException('fixture account details'), ConnectionError('fixture')):
            with self.subTest(error=error):
                with patch('calendar_api.ICloudService', side_effect=error), redirect_stdout(io.StringIO()) as output:
                    calendar = ICloudCalendar()
                self.assertIsNone(calendar.service)
                self.assertNotIn('fixture account details', output.getvalue())
                self.assertIn('error', calendar.dispatch('calendar_next_event'))

    def test_missing_credentials_only_disables_calendar(self):
        with patch.dict('os.environ', {}, clear=True), patch('icloud_service.load_dotenv'), redirect_stdout(io.StringIO()):
            calendar = ICloudCalendar()
        self.assertIn('APPLE_ID or ICLOUD_PWD', calendar.dispatch('calendar_next_event')['error'])

    def test_unexpected_programming_errors_still_surface(self):
        with patch('calendar_api.ICloudService', side_effect=TypeError('fixture bug')):
            with self.assertRaisesRegex(TypeError, 'fixture bug'):
                ICloudCalendar()

    def test_available_calendar_and_later_connection_failure(self):
        service = Mock()
        service.next_event.return_value = {'title': 'fixture event'}
        calendar = ICloudCalendar(service)
        self.assertEqual(calendar.dispatch('calendar_next_event'), {'title': 'fixture event'})
        service.next_event.side_effect = ConnectionError('fixture')
        self.assertIn('No calendar data', calendar.dispatch('calendar_next_event')['error'])

    def test_hal_does_not_display_an_error_as_an_empty_schedule(self):
        source = ast.parse((SRC / 'hal.py').read_text())
        function = next(node for node in source.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'handle_api_call')
        backend = Mock()
        backend.dispatch.return_value = {'error': 'iCloud Calendar is unavailable.'}
        display = Mock()
        namespace = {'calendar_backend': backend, 'display': display, 'json': json, 'logger': Mock()}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SRC / 'hal.py'), 'exec'), namespace)
        result = namespace['handle_api_call']('calendar_next_event', [], 'What is next?')
        self.assertEqual(json.loads(result), backend.dispatch.return_value)
        display.calendar.assert_not_called()


if __name__ == '__main__':
    unittest.main()
