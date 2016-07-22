#!/usr/bin/python

import csv
from pprint import pprint

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
        # but don't bother taking the square root
        # so we use the square of the threshold
        # print(diameter_squared**0.5)
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
        self.quadrant = int(2*self.avgx/1920) + 2*int(2*self.avgy/1280)

    def as_dict(self):
        pass


if __name__ == '__main__':
    with open('../test-data/subject-998.tsv') as csvfile:
        reader = csv.DictReader(csvfile, dialect='excel-tab')

        # get to the first sample data
        row = {'timestamp': 'MSG'}
        while (row['timestamp'] == 'MSG') or (float(row['avgx']) == 0):
            # Skip past messages and blinks. Might be nice to abstract
            # this code into a method of a Sample class.
            row = reader.next()
            continue

        fixations = []
        current_fix = Fixation(row)

        for row in reader:
            if (row['timestamp'] == 'MSG') or (float(row['avgx']) == 0):
                # this is a message or a blink, so ignore
                continue
            elif current_fix.still_going(row):
                current_fix.samples.append(row)
            else:
                current_fix.finish()
                if current_fix.duration >= current_fix.min_duration:
                    fixations.append(current_fix)
                # if it's too short, it gets overwritten
                current_fix = Fixation(row)

        pprint([(f.start_time, f.duration, f.quadrant, f.avgx, f.avgy) for f in fixations])
