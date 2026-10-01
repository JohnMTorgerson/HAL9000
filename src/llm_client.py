import logging
import subprocess
from datetime import datetime
from hal_persona_prompt import prompt as HAL_PERSONA_PROMPT
from followup import FOLLOWUP_FORMAT, FOLLOWUP_INSTRUCTIONS, FollowupDecision


# For OpenAI v1+ usage
try:
    from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
except ImportError:
    OpenAI = None  # Ollama mode won't use this
    OPENAI_ERRORS = ()
else:
    OPENAI_ERRORS = (APIError,)


class LLMServiceError(RuntimeError):
    """An actionable service failure; HAL can return to listening safely."""


def _openai_service_error(error):
    code = getattr(error, 'code', None)
    status = getattr(error, 'status_code', None)
    billing = 'https://platform.openai.com/settings/organization/billing/'
    if code == 'credit_balance_exhausted':
        message = f'OpenAI API credits are exhausted. Add credits at {billing}, then try again.'
    elif code in ('organization_spend_limit_exceeded', 'project_spend_limit_exceeded',
                  'organization_usage_limit_exceeded'):
        message = 'OpenAI API usage is blocked by an account limit. Check the API billing and limit settings.'
    elif code == 'insufficient_quota' or getattr(error, 'type', None) == 'insufficient_quota':
        message = f'OpenAI API quota is unavailable. Check credits and limits at {billing}.'
    elif status == 429:
        message = 'OpenAI is temporarily rate limiting requests. Please wait before asking again.'
    elif status == 401:
        message = 'OpenAI rejected the API key. Check OPENAI_API_KEY and restart HAL.'
    elif status in (400, 403, 404) and getattr(error, 'param', None) == 'service_tier':
        message = ('OpenAI rejected LLM_SERVICE_TIER. Check service tier access for this '
                   'model and project, or set LLM_SERVICE_TIER=default and restart HAL.')
    elif status in (403, 404):
        message = 'OpenAI denied access to the requested resource. Check LLM_MODEL and API key permissions.'
    elif isinstance(error, APITimeoutError):
        message = 'The OpenAI request timed out. Please try again shortly.'
    elif isinstance(error, APIConnectionError):
        message = 'Unable to connect to OpenAI. Check the network connection and try again.'
    elif status is not None and status >= 500:
        message = 'OpenAI is temporarily unavailable. Please try again shortly.'
    else:
        # Do not echo raw API errors: they can contain credentials or user input.
        message = f'OpenAI could not complete the request (HTTP {status or "unknown"}). Check the model and API settings.'
    return LLMServiceError(message)

def get_hal_system_message():
    return {"role": "system", "content": HAL_PERSONA_PROMPT}


class LLMClient:
    def __init__(self, backend, model_name, max_history=6, openai_api_key=None,
                 service_tier=None, logger=None):
        self.backend = backend
        self.model_name = model_name
        self.max_history = max_history
        self.chat_history = []
        self.logger = logger if logger is not None else logging.getLogger('HAL')

        if backend == "openai":
            self.service_tier = (service_tier or '').strip().lower() or None
            if self.service_tier not in (None, 'auto', 'default', 'fast', 'priority'):
                raise ValueError('LLM_SERVICE_TIER must be auto, default, fast, or priority.')
            if OpenAI is None:
                raise ImportError("OpenAI package not found. Please install openai>=1.0.0")
            if not openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI backend")
            # A billing failure cannot recover through retries. Make one attempt
            # per request and return control to the listener on service errors.
            self.client = OpenAI(api_key=openai_api_key, max_retries=0)
            self.logger.info('LLM ready: %s; requested service tier: %s.',
                             self.model_name, self.service_tier or 'auto (project default)')

    def _get_timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _history_with(self, user_input):
        # Commit history only after success; a failed request must not leave an
        # unanswered user turn or discard older conversation through trimming.
        history = self.chat_history + [{"role": "user", "content": f"[{self._get_timestamp()}] {user_input}"}]
        if len(history) > self.max_history * 2:
            history = history[-self.max_history * 2 :]
        return history

    def _openai_response(self, history, *, followup=False, explicitly_addressed=False):
        system_message = get_hal_system_message()
        options = {}
        if followup:
            system_message['content'] += ('\n' + FOLLOWUP_INSTRUCTIONS +
                f'\nApplication signal: explicitly_addressed={str(explicitly_addressed).lower()}.')
            options['response_format'] = FOLLOWUP_FORMAT
        if self.service_tier is not None:
            # The priority alias also works with HAL's pinned OpenAI SDK.
            options['service_tier'] = ('priority' if self.service_tier == 'fast'
                                       else self.service_tier)
        if self.model_name == 'gpt-6-luna':
            options['reasoning_effort'] = 'none'

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[system_message] + history,
                max_completion_tokens=512,
                temperature=1,
                **options,
            )
        except OPENAI_ERRORS as exc:
            raise _openai_service_error(exc) from exc
        self.logger.info('LLM service tier: requested=%s; used=%s.',
                         self.service_tier or 'auto (project default)',
                         getattr(response, 'service_tier', None) or 'not reported')
        return response

    def get_followup_response(self, user_input, *, explicitly_addressed=False):
        """One decision/response request; only accepted turns enter history."""
        if self.backend != 'openai':
            raise LLMServiceError('Follow-up filtering requires LLM_BACKEND=openai.')
        history = self._history_with(user_input)
        response = self._openai_response(history, followup=True,
                                         explicitly_addressed=explicitly_addressed)
        if not response.choices:
            raise LLMServiceError('No follow-up decision received; wake phrase required again.')
        choice = response.choices[0]
        if choice.finish_reason != 'stop' or getattr(choice.message, 'refusal', None):
            raise LLMServiceError('Incomplete follow-up decision; wake phrase required again.')
        try:
            result = FollowupDecision.parse(choice.message.content)
        except ValueError:
            raise LLMServiceError('Invalid follow-up decision; wake phrase required again.') from None
        if result.decision == 'respond':
            self.chat_history = history + [{'role': 'assistant', 'content': result.reply}]
        return result

    def get_response(self, user_input):
        history = self._history_with(user_input)
        if self.backend == "openai":
            response = self._openai_response(history)
            reply = response.choices[0].message.content.strip()
            self.chat_history = history + [{"role": "assistant", "content": reply}]
            return reply

        elif self.backend == "ollama":
            prompt = HAL_PERSONA_PROMPT + "\n" + "\n".join(
                f"{entry['role'].capitalize()}: {entry['content']}" for entry in history
            ) + "\nHAL:"

            try:
                result = subprocess.run(
                    ["ollama", "run", self.model_name, prompt],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                reply = result.stdout.strip()
            except subprocess.CalledProcessError as e:
                reply = f"Error calling Ollama: {e}"

            # Append assistant reply
            self.chat_history = history + [{"role": "assistant", "content": reply}]
            return reply

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
