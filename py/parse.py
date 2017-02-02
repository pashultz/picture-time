#!/usr/bin/env python2

import csv
from itertools import ifilterfalse
import os
import re


class Epoch:
    """A short segment of a trial."""

    def __init__(self, fixations, start_time, duration, image_categories):
        self.start_time, self.duration = start_time, duration
        self.fixations = fixations
        self.calculate_dwell_times()
        self.image_categories = image_categories

    def __repr__(self):
        return(str(self.start_time))

    def calculate_dwell_times(self):
        """How long did the subject dwell on each type of stimulus?"""

        self.dwell_times = {}

        for f in self.fixations:
            # figure out how much time is within the epoch,
            crop = (
                min(self.start_time + self.duration, f.end_time)
                - max(self.start_time, f.start_time)
                )
            # and add the time to the right category
            try:
                self.dwell_times[f.category] += crop
            except KeyError:
                self.dwell_times[f.category] = crop


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
            t.dwell_times = t.aggregate_gaze_data()
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
                        current_fix.classify(trial.neutral_on_right,
                                             trial.image_categories)
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

    def __init__(self, sample, threshold=55,
                 min_duration=100):
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

    def classify(self, neutral_on_right, image_categories,
                 left_boundaries=[662, 520, 362, 120],
                 right_boundaries=[662, 1160, 362, 760]):
        """Set self.category to a string naming the type of image.
        The 'x_boundaries' parameters are lists of coordinates in CSS order:
        [top, right, bottom, left]
        """

        # figure out what part of the screen we're looking at
        if (
                self.avgx < left_boundaries[3]
                or self.avgx > right_boundaries[1]
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

        # TODO this is very fragile; could break with new filenames or
        # OpenSesame variable names.
        if (
            (position == 'left' and neutral_on_right)
            or (position == 'right' and not neutral_on_right)
        ):
            self.category = image_categories[1]  # ewwwwww
        elif (
            (position == 'left' and not neutral_on_right)
            or (position == 'right' and neutral_on_right)
        ):
            self.category = image_categories[0]  # just, no
        else:
            self.category = 'Away'

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
        self.image_categories = self.get_image_categories()

        # which way are the images arranged? True/False
        self.neutral_on_right = (
            self.run_order['LeftImage'] == '[DisgustImage]')

    def __repr__(self):
        """Give a more informative description."""
        return "trial: {}, {} and {}".format(str(self.start_time),
                                             self.run_order['LeftImage'],
                                             self.run_order['RightImage'])

    def get_image_categories(self):
        """Figure out what types of images are being used."""

        # Make a list of things that look like filenames, shorn of
        # their numerical index and extension.

        # This is really fragile, but it's the best way I can see given the
        # data we already have.

        return [re.search(r'[^0-9]+', v).group()
                for v in self.run_order.values()
                if v[-4:] == '.bmp']

    def aggregate_gaze_data(self, resolution=(1280, 1024)):
        """Returns a dictionary: {'category': time, ... } """

        dwell_times = {}
        for f in self.fixations:
            try:
                dwell_times[f.category] += f.duration
            except KeyError:
                dwell_times[f.category] = f.duration

        return dwell_times

    def make_epochs(self, duration=500):
        """Divide data into epochs of default length 500ms."""

        self.epochs = []

        for t in range(self.start_time, self.end_time-duration, duration):
            fixes = [f for f in self.fixations if
                     (f.end_time > t and f.start_time < t + duration)]
            self.epochs.append(
                Epoch(fixations=fixes,
                      start_time=t,
                      duration=duration,
                      image_categories=self.image_categories
                      )
            )

        # don't forget the last one

    def orienting_category(self):
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

        if self.neutral_on_right:
            sides = self.image_categories[::-1]
        else:
            sides = self.image_categories

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
def orienting_bias(trials):
    """Given a list of trials, return the proportion (0<x<1) that begin
    with a fixation on the disgusting stimulus.
    """

    orients = [t.orienting_category() for t in trials
               if t.orienting_category]

    if len(orients) == 0:
        return None
    else:
        return ((orients.count('Poop') +
                orients.count('Dog'))
                / float(len(orients)))


def get_subject_numbers(
        directory='/home/pashultz/Dropbox/disgust-habituation/experiment/data/'
        ):
    return sorted([int(f[f.index('-') + 1:f.index('.')])
                   for f in
                   os.listdir(directory)
                   if f[-3:] == 'tsv'])


def tabulate_epoch_statistics(subject_numbers):
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
                times_d.append(current.dwell_times['Poop']
                               if 'Poop' in current.dwell_times else 0)
                times_n.append(current.dwell_times['Colors']
                               if 'Colors' in current.dwell_times else 0)
                times_a.append(current.dwell_times['Away']
                               if 'Away' in current.dwell_times else 0)
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


def tabulate_orienting_bias(subject_numbers):
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

    for subject in subject_numbers:
        print('trying {}'.format(subject))
        e = Experiment(subject)
        biases.append({
            'subject': subject,
            'bias_block1': orienting_bias(e.trials[:24]),
            'bias_block2': orienting_bias(e.trials[25:])
            })

        for i in range(6):
            (biases[-1]["bias_block1_{}".format(i + 1)]
             ) = orienting_bias(e.subblocks[i])
        for i in range(6):
            (biases[-1]["bias_block2_{}".format(i + 1)]
             ) = orienting_bias(e.subblocks[i + 6])

    return biases


def tabulate_dwells_per_subject(subject_numbers):
    """Returns a list of dicts, with each trial keyed as d|n[block].[trial]."""

    # This code pulls out variables and puts them in SPSS-firendly format
    subjects = []
    for s in [sub for sub in subject_numbers if sub <= 414]:
        print('loading subject {}'.format(s))
        e = Experiment(s)
        d = {'asub': e.subject_number}
        try:
            d.update({
                'd1.{:02}'.format(t + 1):
                (e.trials[t].dwell_times['Poop']
                 if 'Poop' in e.trials[t].dwell_times else 0)
                for t in range(24)})
            d.update({
                'd2.{:02}'.format(t - 23):
                (e.trials[t].dwell_times['Poop']
                 if 'Poop' in e.trials[t].dwell_times else 0)
                for t in range(24, 48)})
            d.update({
                'n1.{:02}'.format(t + 1):
                (e.trials[t].dwell_times['Colors']
                 if 'Colors' in e.trials[t].dwell_times else 0)
                for t in range(24)})
            d.update({
                'n2.{:02}'.format(t - 23):
                (e.trials[t].dwell_times['Colors']
                 if 'Colors' in e.trials[t].dwell_times else 0)
                for t in range(24, 48)})
            d.update({
                'w1.{:02}'.format(t + 1):
                (e.trials[t].dwell_times['Away']
                 if 'Away' in e.trials[t].dwell_times
                 else 0)
                for t in range(24)})
            d.update({
                'w2.{:02}'.format(t - 23):
                (e.trials[t].dwell_times['Away']
                 if 'Away' in e.trials[t].dwell_times
                 else 0)
                for t in range(24, 48)})
            subjects.append(d)
        except IndexError:
            print("Not enough trials")

    return subjects


def collect_fear_stimulus_results(subjects):
    """Return a list of dictionaries corresponding to subjects 500 and up.

    Blocks in random (counterbalanced) order: neutral/disgust, neutral/fear.
    """

    # subjects = [s for s in get_subject_numbers() if s >= 500]
    results = []

    for s in subjects:
        print("loading {}...".format(s))
        e = Experiment(s)
        res = {}
        res = {"subject": e.subject_number}
        for i in range(24):
            t = e.trials[i]
            # it's either poop or a dog
            if 'Poop' in t.image_categories:
                res["d{:02}".format(i+1)] = (
                    t.dwell_times['Poop'] if 'Poop' in t.dwell_times
                    else 0)
            else:
                res["t{:02}".format(i+1)] = (
                    t.dwell_times['Dog'] if 'Dog' in t.dwell_times
                    else 0)
            res["nd{:02}".format(i+1)] = (
                t.dwell_times['Colors'] if 'Colors' in t.dwell_times
                else 0)
            res["wd{:02}".format(i+1)] = (
                t.dwell_times['Away'] if 'Away' in t.dwell_times
                else 0)
        for i in range(24, 48):
            t = e.trials[i]
            if 'Poop' in t.image_categories:
                res["d{:02}".format(i-23)] = (
                    t.dwell_times['Poop'] if 'Poop' in t.dwell_times
                    else 0)
            else:
                res["t{:02}".format(i-23)] = (
                    t.dwell_times['Dog'] if 'Dog' in t.dwell_times
                    else 0)
            res["nt{:02}".format(i-23)] = (
                t.dwell_times['Colors'] if 'Colors' in t.dwell_times
                else 0)
            res["wt{:02}".format(i-23)] = (
                t.dwell_times['Away'] if 'Away' in t.dwell_times
                else 0)

        print(res)
        results.append(res)

    return results


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
    # subjects >= 500 have threat stimuli, not just disgust
    subjects = [s for s in get_subject_numbers() if s >= 500]
    results = tabulate_orienting_bias(subjects)
    write_dictlist_to_csv(results,
                          'orienting_bias_second_condition.csv',
                          '../../disgust-habituation/experiment/data/')
