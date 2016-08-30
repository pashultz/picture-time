import unittest
import parse


class TestExperimentConstruction(unittest.TestCase):

    # def test_count_trials(self):
    #     for i in range(300, 310):
    #         e = parse.Experiment(i)
    #         self.assertEquals(len(e.trials), 48)

    # def test_trial_length(self):
    #     for i in range(300, 303):
    #         e = parse.Experiment(i)
    #         for t in e.trials:
    #             self.assertTrue(11000 < t.end_time - t.start_time < 13000)

    def test_fixation_length(self):
        for i in range(300, 333):
            e = parse.Experiment(i)
            for t in e.trials:
                total_fixations = sum(f.duration for f in t.fixations)
                print("trial {}.{}: {}".format(
                    i,
                    e.trials.index(t),
                    total_fixations))
                self.assertLess(total_fixations, 12000)

if __name__ == '__main__':
    unittest.main()
