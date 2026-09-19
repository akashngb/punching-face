"""Verification tracks code dependencies without executing optional backends."""

from pathlib import Path
import tempfile
import unittest

from scripts.verify_texture_bake import local_code_fingerprint


class CodeFingerprintTests(unittest.TestCase):
    def test_nested_relative_helpers_and_package_initializers_are_tracked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / 'bake'
            package.mkdir()
            (package / '__init__.py').write_text('raise RuntimeError("never import")')
            main = root / 'main.py'
            main.write_text('def run():\n    from bake import appearance\n')
            (package / 'appearance.py').write_text('from . import alignment\n')
            helper = package / 'alignment.py'
            helper.write_text('SUPPORT = 0.1\n')
            before = local_code_fingerprint(root, [main])
            self.assertEqual(
                set(before),
                {
                    'main.py',
                    'bake/__init__.py',
                    'bake/appearance.py',
                    'bake/alignment.py',
                },
            )
            helper.write_text('SUPPORT = 0.2\n')
            after = local_code_fingerprint(root, [main])
            self.assertNotEqual(before['bake/alignment.py'], after['bake/alignment.py'])
            self.assertEqual(before['main.py'], after['main.py'])

    def test_unrelated_changes_and_formatting_do_not_invalidate_bake(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            main = root / 'main.py'
            main.write_text('import helper\nVALUE=1\n')
            (root / 'helper.py').write_text('import main\n')
            unrelated = root / 'unused.py'
            unrelated.write_text('VALUE=0\n')
            before = local_code_fingerprint(root, [main])
            main.write_text('# formatting only\nimport helper\nVALUE = 1\n')
            unrelated.write_text('VALUE=2\n')
            self.assertEqual(before, local_code_fingerprint(root, [main]))

    def test_new_dependency_is_discovered_and_external_libraries_are_not_imported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            main = root / 'main.py'
            main.write_text('import unavailable_optional_gpu_backend\n')
            before = local_code_fingerprint(root, [main])
            self.assertEqual(set(before), {'main.py'})
            main.write_text(
                'import unavailable_optional_gpu_backend\nimport new_helper\n'
            )
            (root / 'new_helper.py').write_text('raise RuntimeError("do not run")\n')
            after = local_code_fingerprint(root, [main])
            self.assertIn('new_helper.py', after)
            self.assertNotEqual(before, after)


if __name__ == '__main__':
    unittest.main()
