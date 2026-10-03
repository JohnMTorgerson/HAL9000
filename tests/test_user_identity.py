"""Configured names flow through persona, cleanup and memory without hardware."""
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from dotenv import load_dotenv

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from conversation_memory import ConversationMemory, MemorySettings
from memory_store import MemoryStore
from user_identity import get_user_name


class UserIdentityTests(unittest.TestCase):
    def test_missing_or_blank_name_uses_existing_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_user_name(), 'Dave')
            for value in ('', '   '):
                os.environ['HAL_USER_NAME'] = value
                self.assertEqual(get_user_name(), 'Dave')

    def test_dotenv_name_reaches_persona_cleanup_and_memory(self):
        # spaCy is unrelated to this helper; avoid loading its NLP model.
        with patch.dict(sys.modules, {'spacy': Mock()}):
            strip_name = runpy.run_path(str(SRC / 'helper_funcs.py'))['strip_name_at_sentence_end']
        for name in ('Alex McKenzie', 'Renée', "O'Neill"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / '.env'
                config.write_text(f'HAL_USER_NAME="  {name}  "\n')
                with patch.dict(os.environ, {}, clear=True):
                    load_dotenv(config)
                    self.assertEqual(get_user_name(), name)
                    persona = runpy.run_path(str(SRC / 'hal_persona_prompt.py'))
                    self.assertEqual(persona['USER'], name)
                    self.assertIn(f"address the user as '{name}'", persona['prompt'])
                    self.assertEqual(strip_name(f'The task is complete, {name}.'), 'The task is complete.')
                    question = f'Are you ready, {name}?'
                    self.assertEqual(strip_name(question), question)
                    self.assertEqual(strip_name('The task is complete, Sam.', name='Sam'), 'The task is complete.')
                    with tempfile.TemporaryDirectory() as data:
                        store = MemoryStore(data, logger=Mock())
                        memory = ConversationMemory(store, Mock(), MemorySettings(), Mock(), Mock())
                        try:
                            _, context = memory.read_context()
                            self.assertIn(f'about {name}, the single user', context)
                            self.assertNotIn('{user_name}', context)
                        finally:
                            memory.close()


if __name__ == '__main__':
    unittest.main()
