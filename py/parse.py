#!/usr/bin/env python2

import csv
from itertools import ifilterfalse
import os
import re


class Epoch:
    """A short segment of a trial."""

    def __init__(self, fixations, start_time, duration):
        self.start_time, self.duration = start_time, duration
        self.fixations = fixations
        self.calculate_dwell_times()

    def __repr__(self):
        return(str(self.start_time))

    def calculate_dwell_times(self):
        """How long did the subject dwell on each type of stimulus?"""

        self.time_away = self.time_disgust = self.time_neutral = 0

        for f in self.fixations:
            # figure out how much time is within the epoch,
            crop = (
                min(self.start_time + self.duration, f.end_time)
                - max(self.start_time, f.start_time)
                )

            # and add the time to the right category
            if f.category == 'away':
                self.time_away += crop
            elif f.category == 'disgust':
                self.time_disgust += crop
            elif f.category == 'neutral':
                self.time_neutral += crop
            else:
                raise(Exception('Unclassified fixation'))


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
        # two blocks of six subblocks of four trials each
        self.subblocks = [self.trials[4*i:4*(i+1)] for i in range(12)]

        self.get_fixations()

        for t in self.trials:
            (t.time_disgust,
             t.time_neutral,
             t.time_away) = t.aggregate_gaze_data()
            t.make_epochs()

    def get_fixations(self):
        """Populates the trials with lists of fixations.

        It may be inelegant to do this at the level of the Experiment
        rather than the Trial, but my first attempt at
        Trial.get_fixations() involved scanning the entire .tsv file
        48 times for each experiment, which made things unacceptably
        slow.
        """

        with open(self.directory
                  + "/subject-" + str(self.subject_number)
                  + ".tsv") as tsvfile:
            reader = csv.DictReader(tsvfile, dialect='excel-tab')

            # filter out messages
            rows = ifilterfalse(
                (lambda row: row['timestamp'] == 'MSG'),
                reader
            )

            for i in range(len(self.trials)):
                trial = self.trials[i]
                trial.fixations = []

                # get to the beginning of the trial
                row = rows.next()
                while int(row['time']) < trial.start_time:
                    row = rows.next()

                # try a fixation, just to get things going
                current_fix = Fixation(row)
                row = rows.next()

                while int(row['time']) < trial.end_time:
                    if current_fix.is_still_going(row):
                        current_fix.samples.append(row)
                        row = rows.next()
                    else:
                        current_fix.finish()
                        current_fix.classify(trial.disgust_on_left)
                        if current_fix.duration >= current_fix.min_duration:
                            trial.fixations.append(current_fix)
                        # if it's too short, just overwrite it with a new one
                        current_fix = Fixation(row)
                        row = rows.next()

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

    is_still_going(dict) -- decide whether a sample continues this
        fixation event.

    finish() -- calculate a bunch of properties when the fixation is done.

    as_dict() -- Dump the data as a dictionary:
        'trial' (int): number of trial
        # 'ordinal' (int): ordinal number of fixation within the trial
        #     wait, actually we can just get this from the list of fixations
        'start_time' (int): time of start
        'end_time' (int): time of end
        'duration' (int): duration in ms
        'avgx' (float): average x for all samples within fixation
        'avgy' (float): average y
        'category' (string): the type of image being stared at
    """

    def __init__(self, sample, threshold=55, min_duration=100):
        self.start_time = int(sample['time'])
        self.samples = [sample]
        # the dispersion window starts as a point
        self.max_x, self.max_y = float(sample['avgx']), float(sample['avgy'])
        self.min_x, self.min_y = self.max_x, self.max_y
        # the dispersion threshold and required duration
        self.threshold_squared = threshold**2
        self.min_duration = min_duration

    def __repr__(self):
        """Show a more informative description."""

        return("fixation at {} dur {} {}".format(self.start_time,
                                                 self.duration,
                                                 self.category))

    def classify(self, disgust_on_left,
                 left_boundaries=[662, 520, 362, 120],
                 right_boundaries=[662, 1160, 362, 760]):
        """Set self.category to a string naming the type of image.
        The 'x_boundaries' parameters are lists of coordinates in CSS order:
        [top, right, bottom, left]
        """

        # figure out what part of the screen we're looking at
        if (
                self.avgx < left_boundaries[3]
                or self.avgx > right_boundaries[3]
                or self.avgy < left_boundaries[2]
                or self.avgy > left_boundaries[0]
        ):
            position = 'out of bounds'
        elif left_boundaries[1] < self.avgx < right_boundaries[3]:
            position = 'in between'
        elif self.avgx < left_boundaries[1]:
            position = 'left'
        else:
            position = 'right'

        # figure out what category that corresponds to
        if (
            (position == 'left' and disgust_on_left)
            or (position == 'right' and not disgust_on_left)
        ):
            self.category = 'disgust'
        elif (
            (position == 'left' and not disgust_on_left)
            or (position == 'right' and disgust_on_left)
        ):
            self.category = 'neutral'
        else:
            self.category = 'away'

    def finish(self):
        # if the fixation is only a blink event
        if not hasattr(self, 'samples'):
            self.duration = 0
            return

        self.end_time = int(self.samples[-1]['time'])
        self.duration = self.end_time - self.start_time
        self.avgx = (sum([float(s['avgx']) for s in self.samples]) /
                     len(self.samples))
        self.avgy = (sum([float(s['avgy']) for s in self.samples]) /
                     len(self.samples))
        # quadrants are numbered l2r, t2b
        # this seemed like a good way to calculate them, but maybe not...
        # TODO pass screen resolution as parameters
        self.quadrant = int(2*self.avgx/1280) + 2*int(2*self.avgy/1024)
        # stop it from running out of memory!
        del self.samples

    def is_still_going(self, sample):
        """Decides whether the sample continues this fixation."""

        # if there's a blink, scrap the fixation
        if sample['avgx'] == '0.0':
            return False

        # update the dispersion boundary
        self.max_x = max(self.max_x, float(sample['avgx']))
        self.max_y = max(self.max_y, float(sample['avgy']))
        self.min_x = min(self.min_x, float(sample['avgx']))
        self.min_y = min(self.min_y, float(sample['avgy']))
        diameter_squared = (
            (self.max_x - self.min_x)**2
            + (self.max_y - self.min_y)**2
            )

        # compare the diameter of our dispersion window to the threshold
        # but we use the square of the threshold to save a step
        # by not taking the square root
        if diameter_squared < self.threshold_squared:
            return True
        else:
            return False


class Trial:
    """An image-viewing trial."""
    def __init__(self, experiment, start_time, end_time, run_order):
        self.start_time = start_time
        self.end_time = end_time
        self.run_order = run_order

        # which way are the images arranged? True/False
        self.disgust_on_left = (
            self.run_order['LeftImage'] == '[DisgustImage]')

    def __repr__(self):
        """Give a more informative description."""
        return "trial: {}, {} and {}".format(str(self.start_time),
                                             self.run_order['LeftImage'],
                                             self.run_order['RightImage'])

    def aggregate_gaze_data(self, resolution=(1280, 1024)):
        """Returns a tuple: (time_disgust, time_neutral, time_away)"""

        # This is the old way. It worked, but now the fixations are
        # already classified.
        ######
        # time_left = sum(f.duration
        #                 for f in self.fixations
        #                 # let's do this the quick-and-dirty way
        #                 if 120 <= f.avgx <= 520 and 362 <= f.avgy <= 662
        #                 )
        # time_right = sum(f.duration
        #                  for f in self.fixations
        #                  if 760 <= f.avgx <= 1160 and 362 <= f.avgy <= 662
        #                  )
        # time_away = sum(f.duration
        #                 for f in self.fixations
        #                 if (
        #                     f.avgx <= 120 or f.avgx >= 1160 or
        #                     f.avgy <= 362 or f.avgy >= 662 or
        #                     520 <= f.avgx <= 760
        #                    )
        #                 )

        time_disgust = time_neutral = time_away = 0

        for f in self.fixations:
            if f.category == 'disgust':
                time_disgust += f.duration
            elif f.category == 'neutral':
                time_neutral += f.duration
            elif f.category == 'away':
                time_away += f.duration
            else:
                raise(Exception('Unclassified fixation'))

        return(time_disgust, time_neutral, time_away)

    def make_epochs(self, duration=500):
        """Divide data into epochs of default length 500ms."""

        self.epochs = []

        for t in range(self.start_time, self.end_time-duration, duration):
            fixes = [f for f in self.fixations if
                     (f.end_time > t and f.start_time < t + duration)]
            self.epochs.append(Epoch(fixes, t, duration))

        # don't forget the last one

    def orientation_category(self):
        """After an initial fixation at center, the category of the first
        image the subject fixes on. If the first fixation isn't
        centered, or if it's wonky for some other reason, raise an
        InvalidTrial exception.
        """

        # the center of the screen
        # TODO un-hardcode the resolution
        center = complex(1280/2, 1024/2)

        # the offset of each fixation from center
        offsets = [complex(f.avgx, f.avgy)
                   - center for f in self.fixations]

        # if the first fixation isn't in the center, chuck the trial
        try:
            # TODO un-hardcode the centering threshold
            if abs(offsets[0]) > 55:
                return None
        # also if there are no fixations
        except IndexError:
            return None

        # TODO maybe a good use for a custom exception?
            # raise InvalidTrial(
            #     'Fixation not centered in trial starting at {}'
            #     .format(self.start_time))

        noncentered_fixations = filter(lambda x: abs(x) > 55, offsets)
        try:
            bias = noncentered_fixations[0].real
        except IndexError:
            # if no noncentered fixations, chuck it
            return None

        if self.disgust_on_left:
            sides = ('disgust', 'neutral')
        else:
            sides = ('neutral', 'disgust')

        if bias < 0:
            return sides[0]
        else:
            return sides[1]


# User-defined exceptions go here
class InvalidTrial(Exception):
    """Something has gone wrong with the trial"""

    def __init__(self, value=''):
        self.value = value

    def __str__(self):
        return repr(self.value)


# Global functions
def orientation_bias(trials):
    """Given a list of trials, return the proportion (0<x<1) that begin
    with a fixation on the disgusting stimulus.
    """

    orients = [t.orientation_category() for t in trials
               if t.orientation_category]

    if len(orients) == 0:
        return None
    else:
        return orients.count('disgust')/float(len(orients))


def tabulate_epoch_statistics(
        directory='../../disgust-habituation/experiment/data'):
    """Find the dwell time for each stimulus in each epoch, averaged
    across all trials.

    returns a list of dictionaries:
    {
        "subject": subject number,
        "e01_d": epoch 1 disgust,
        "e01_n": epoch 1 neutral,
        "e01_a": epoch 1 away,
        "e02_d": epoch 2 disgust,
        etc.
    }
    """

    subject_numbers = sorted([int(f[f.index('-') + 1:f.index('.')])
                              for f in
                              os.listdir(directory)
                              if f[-3:] == 'tsv'])

    results = []

    for subject in subject_numbers:
        print('trying experiment {}...'.format(subject))
        exp = Experiment(subject)
        results.append({'subject': subject})

        # list of averages for each epoch across all trials
        averages_disgust = []
        averages_neutral = []
        averages_away = []
        for ep in range(0, 24):
            # for each epoch
            # lists of times for this epoch in all trials
            times_d = []
            times_n = []
            times_a = []
            for tr in range(0, len(exp.trials)):
                # for each trial
                # note that not all subjects have the full 48 trials, e.g. 353
                current = exp.trials[tr].epochs[ep]
                times_d.append(current.time_disgust)
                times_n.append(current.time_neutral)
                times_a.append(current.time_away)
            averages_disgust.append(sum(times_d)/float(len(times_d)))
            averages_neutral.append(sum(times_n)/float(len(times_n)))
            averages_away.append(sum(times_a)/float(len(times_a)))
            results[-1].update({
                'e{:02}_d'.format(ep + 1): sum(times_d)/float(len(times_d)),
                'e{:02}_n'.format(ep + 1): sum(times_n)/float(len(times_n)),
                'e{:02}_a'.format(ep + 1): sum(times_a)/float(len(times_a))
                })
            # print(results[-1])

    return results


def tabulate_orientation_bias(directory):
    """Returns a list of dicts. For each subject, it gives:
    - subject
    - bias_block1
    - bias_block1_1
    - bias_block1_2
    - bias_block1_3
    - bias_block1_4
    - bias_block1_5
    - bias_block1_6
    - bias_block2
    - bias_block2_1
    - bias_block2_2
    - bias_block2_3
    - bias_block2_4
    - bias_block2_5
    - bias_block2_6
    """

    biases = []

    subject_numbers = sorted([int(f[f.index('-') + 1:f.index('.')])
                              for f in
                              os.listdir(directory)
                              if f[-3:] == 'tsv'])

    for subject in subject_numbers:
        print('trying {}'.format(subject))
        e = Experiment(subject)
        biases.append({
            'subject': subject,
            'bias_block1': orientation_bias(e.trials[:24]),
            'bias_block2': orientation_bias(e.trials[25:])
            })

        for i in range(6):
            (biases[-1]["bias_block1_{}".format(i + 1)]
             ) = orientation_bias(e.subblocks[i])
        for i in range(6):
            (biases[-1]["bias_block2_{}".format(i + 1)]
             ) = orientation_bias(e.subblocks[i + 6])

    return biases


def tabulate_trials_per_subject(directory):
    """Returns a list of dicts, with each trial keyed as d|n[block].[trial]."""

    subject_numbers = sorted([int(f[f.index('-') + 1:f.index('.')])
                              for f in
                              os.listdir(directory)
                              if f[-3:] == 'tsv'])
    # This code pulls out variables and puts them in SPSS-firendly format
    subjects = []
    for s in [sub for sub in subject_numbers if sub <= 414]:
        print('loading subject {}'.format(s))
        e = Experiment(s, directory)
        d = {'asub': e.subject_number}
        try:
            d.update({'d1.{:02}'.format(t + 1): e.trials[t].time_disgust
                      for t in range(24)})
            d.update({'d2.{:02}'.format(t - 23): e.trials[t].time_disgust
                      for t in range(24, 48)})
            d.update({'n1.{:02}'.format(t + 1): e.trials[t].time_neutral
                      for t in range(24)})
            d.update({'n2.{:02}'.format(t - 23): e.trials[t].time_neutral
                      for t in range(24, 48)})
            d.update({'w1.{:02}'.format(t + 1): e.trials[t].time_away
                      for t in range(24)})
            d.update({'w2.{:02}'.format(t - 23): e.trials[t].time_away
                      for t in range(24, 48)})
            subjects.append(d)
        except IndexError:
            print("Not enough trials")

    return subjects


# This code writes the data file
def write_dictlist_to_csv(dictlist, filename, directory):
    """Write a list of dicts to a CSV file."""

    with open(directory + filename, 'w') as f:
        writer = csv.DictWriter(f,
                                sorted(dictlist[0].keys()))
        writer.writeheader()
        for d in dictlist:
            writer.writerow(d)


if __name__ == '__main__':
    subjects = tabulate_epoch_statistics()
    write_dictlist_to_csv(subjects,
                          'epochs_by_subject.csv',
                          '../../disgust-habituation/experiment/data/')
