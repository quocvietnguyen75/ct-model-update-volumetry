"""Synthetic contracts for the manuscript build script's figure copies (scripts/ct_manuscript_latex.py)."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/ct_manuscript_latex.py'
EXAMPLE = 'figure4_example_pair.pdf'


def load_script():
    spec = importlib.util.spec_from_file_location('ct_manuscript_latex_script', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FigureCopyTests(unittest.TestCase):
    def setUp(self):
        self.script = load_script()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.source, self.target = Path(folder.name)/'print', Path(folder.name)/'latex'
        self.source.mkdir()
        self.target.mkdir()

    def draw(self, skip=()):
        for name, _ in self.script.FIGURE_COPIES:
            if name not in skip:
                (self.source/name).write_bytes(name.encode())

    def test_every_figure_is_copied_under_its_manuscript_name(self):
        self.draw()
        written = self.script.copy_figures(self.source, self.target)
        self.assertEqual(written, ['figure1.pdf', 'figure2.pdf', 'figure3.pdf', 'figure4.pdf', 'figureS1.pdf',
                                   'figureS2.pdf'])
        for name, target in self.script.FIGURE_COPIES:
            self.assertEqual((self.target/target).read_bytes(), name.encode())

    def test_only_the_example_figure_may_be_absent_and_is_not_counted(self):
        self.draw(skip={EXAMPLE})
        written = self.script.copy_figures(self.source, self.target)
        self.assertEqual(written, ['figure1.pdf', 'figure2.pdf', 'figure3.pdf', 'figureS1.pdf', 'figureS2.pdf'])
        self.assertFalse((self.target/'figure4.pdf').exists())

    def test_a_missing_figure_drawn_from_the_records_is_an_error(self):
        self.draw(skip={'figure2_replication_runs.pdf'})
        with self.assertRaises(FileNotFoundError):
            self.script.copy_figures(self.source, self.target)


if __name__ == '__main__':
    unittest.main()
