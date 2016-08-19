#!/usr/bin/python

import csv
from itertools import ifilterfalse
from pprint import pprint
import re
import sys


class Experiment:
    """A run of the experiment."""

    def __init__(self, subject_number,
                 directory='../../disgust-habituation/experiment/data'):
        self.subject_number = subject_number
        self.directory = directory

        self.blocks = 2
        # TODO get resolution from the CSV
        self.resolution = (1280, 1024)
        self.timestamp_offset = self.get_timestamp_offset()
        self.experiment_start_time = self.get_experiment_start_time()
        self.trials = self.get_trials()

        self.get_fixations()

        for t in self.trials:
            t.time_disgust, t.time_neutral = t.aggregate_gaze_data()

    def get_fixations(self):
        """Populates the trials with lists of fixations."""

        with open(self.directory
                  + "/subject-" + str(self.subject_number)
                  + ".tsv") as tsvfile:
            reader = csv.DictReader(tsvfile, dialect='excel-tab')

            # filter out messages and blinks
            # and ignore everything before and after this trial

            rows = ifilterfalse(
                (lambda row: row['timestamp'] == 'MSG'
                 or row['avgx'] == '0.0'),
                reader
            )

            for i in range(len(self.trials)):
                print('processing trial {}.{}'.format(self.subject_number, i))
                trial = self.trials[i]
                trial.fixations = []
                try:
                    current_fix = Fixation(rows.next())
                except StopIteration:
                    print('Warning: no samples '
                          'for Subject {} starting at {}'
                          .format(self.subject_number, trial.start_time))
                    continue

                row = rows.next()
                while int(row['time']) < trial.end_time:
                    if current_fix.still_going(row):
                        current_fix.samples.append(row)
                    else:
                        current_fix.finish()
                        if current_fix.duration >= current_fix.min_duration:
                            trial.fixations.append(current_fix)
                            # if it's too short, it gets overwritten
                        current_fix = Fixation(row)
                    try:
                        row = rows.next()
                    except StopIteration:
                        break

    def get_trials(self):
        """Returns a list of Trial objects with run-order information."""
        with open(self.directory + "/subject-"
                  + str(self.subject_number) + ".csv") as csvfile:
            # get the run order info for the experiment
            reader = csv.DictReader(csvfile, dialect='excel')

            trials = []
            oldrow = reader.next()
            for row in reader:
                # check whether this row is a trial
                if row['time_image_array'] == oldrow['time_image_array']:
                    oldrow = row
                    continue
                else:
                    trial_start = (
                        int(row['time_image_array'])
                        + self.timestamp_offset - 16)
                    trial_end = (
                        int(row['time_InterTrialInterval'])
                        + self.timestamp_offset + 16)
                    # select the data to keep
                    trials.append(Trial(self,
                                        trial_start,
                                        trial_end,
                                        {key: row[key] for key in
                                         ('time_FixDot',
                                          'time_image_array',
                                          'time_InterTrialInterval',
                                          'LeftImage',
                                          'RightImage',
                                          'DisgustImage',
                                          'NeutralImage')}))
                    oldrow = row
            return trials

    def get_timestamp_offset(self):
        """Returns the difference between EyeTribe and OpenSesame timestamps.
        """
        with open(self.directory + "/subject-"
                  + str(self.subject_number) + ".tsv") as tsvfile:
            # figure out the offset between the EyeTribe clock and
            # the OpenSesame timestamps
            p = re.compile('MSG	([^	]+)	'
                           '([0-9]+)	'
                           'var time_new_pygaze_log ([0-9]+)')
            t = p.search(tsvfile.read()).groups()

            # if we ever need a Python datetime, it's
            # datetime.strptime(t[0] + '000', '%Y-%m-%d %H:%M:%S.%f')
            return int(t[1]) - int(t[2])

    def get_experiment_start_time(self):
        """Returns the OpenSesame timestamp for the experiment start."""

        with open(self.directory + "/subject-"
                  + str(self.subject_number) + ".tsv") as tsvfile:
            # figure out the offset between the EyeTribe clock and
            # the OpenSesame timestamps
            p = re.compile('var time_experiment ([0-9]+)')
            t = p.search(tsvfile.read()).group(1)
            return int(t)


class Fixation:
    """A fixation event.

    Takes a dictionary as an argument, corresponding to a sample event.

    Keyword arguments:
    - threshold (int): maximum radius in pixels
    - min_duration (int): minimum duration in ms

    Public methods:

    still_going(dict) -- decide whether a sample continues this fixation event.

    finish() -- calculate a bunch of properties when the fixation is done.

    as_dict() -- Dump the data as a dictionary:
        'trial' (int): number of trial
        # 'ordinal' (int): ordinal number of fixation within the trial
        #     wait, actually we can just get this from the list of fixations
        'start' (int): time of start
        'end' (int): time of end
        'duration' (int): duration in ms
        'avgx' (float): average x for all samples within fixation
        'avgy' (float): average y
        'quadrant' (int): the quadrant of the screen
    """

    def __init__(self, sample, threshold=50, min_duration=100):
        self.start_time = int(sample['time'])
        self.samples = [sample]
        # the dispersion window starts as a point
        self.max_x, self.max_y = float(sample['avgx']), float(sample['avgy'])
        self.min_x, self.min_y = self.max_x, self.max_y
        # the dispersion threshold and required duration
        self.threshold_squared = threshold**2
        self.min_duration = min_duration

    def still_going(self, sample):
        """Decides whether the sample continues this fixation."""

        # update the dispersion boundary
        self.max_x = max(self.max_x, float(sample['avgx']))
        self.max_y = max(self.max_y, float(sample['avgy']))
        self.min_x = min(self.min_x, float(sample['avgx']))
        self.min_y = min(self.min_y, float(sample['avgy']))
        diameter_squared = (self.max_x - self.min_x)**2
        + (self.max_y - self.min_y)**2

        # compare the diameter of our dispersion window to the threshold
        # but we use the square of the threshold to save a step
        # by not taking the square root
        if diameter_squared < self.threshold_squared:
            return True
        else:
            return False

    def finish(self):
        self.end = int(self.samples[-1]['time'])
        self.duration = self.end - self.start_time
        self.avgx = (sum([float(s['avgx']) for s in self.samples]) /
                     len(self.samples))
        self.avgy = (sum([float(s['avgy']) for s in self.samples]) /
                     len(self.samples))
        # quadrants are numbered l2r, t2b
        # this seemed like a good way to calculate them, but maybe not...
        self.quadrant = int(2*self.avgx/1280) + 2*int(2*self.avgy/1024)
        del self.samples


class Trial:
    """An image-viewing trial."""
    def __init__(self, experiment, start_time, end_time, run_order):
        self.start_time = start_time
        self.end_time = end_time
        self.run_order = run_order

        # which way are the images arranged? True/False
        self.disgust_on_left = (
            self.run_order['LeftImage'] == '[DisgustImage]')

    def aggregate_gaze_data(self, resolution=(1280, 1024)):
        """Returns a tuple: (time_disgust, time_neutral)"""

        time_left = sum(f.duration
                        for f in self.fixations
                        if f.avgx < resolution[0] / 2)
        time_right = sum(f.duration
                         for f in self.fixations
                         if f.avgx >= resolution[0] / 2)

        if self.disgust_on_left:
            return (time_left, time_right)
        else:
            return (time_right, time_left)

def tabulate_gaze(directory):
    for i in range(302, 306):
        print('Processing subject {}...'.format(i))
        e = Experiment(i)
        block1, block2 = e.trials[:24], e.trials[24:]
        print('block 1 | disgust: {}ms; neutral: {}ms'.format(
            sum(t.time_disgust for t in block1),
            sum(t.time_neutral for t in block1)))
        print('block 2 | disgust: {}ms; neutral: {}ms'.format(
            sum(t.time_disgust for t in block2),
            sum(t.time_neutral for t in block2)))
        del e


def tabulate_trials_per_subject(directory):
    """Returns a list of dicts, with each trial keyed as d|n{block}.{trial}."""

    subjects = []
    for s in range(300, 333):
        print('loading subject {}'.format(s))
        e = Experiment(s, directory)
        d = {'asub': e.subject_number}
        d.update({'d1.{:02}'.format(t + 1): e.trials[t].time_disgust
                  for t in range(24)})
        d.update({'d2.{:02}'.format(t-23): e.trials[t].time_disgust
                  for t in range(24, 48)})
        d.update({'n1.{:02}'.format(t + 1): e.trials[t].time_neutral
                  for t in range(24)})
        d.update({'n2.{:02}'.format(t-23): e.trials[t].time_neutral
                  for t in range(24, 48)})
        subjects.append(d)

    return subjects


def write_dictlist_to_csv(dictlist, filename, directory):
    """Write a list of dicts to a CSV file."""

    with open(directory + filename, 'w') as f:
        writer = csv.DictWriter(f,
                                sorted(dictlist[0].keys()))
        writer.writeheader()
        for d in dictlist:
            writer.writerow(d)


if __name__ == '__main__':
    subjects = tabulate_trials_per_subject(
        '../../disgust-habituation/experiment/data')
    write_dictlist_to_csv(subjects,
                          'trials_by_subject.csv',
                          '../../disgust-habituation/experiment/data/')
