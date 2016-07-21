#!/usr/bin/python

import csv


class Fixation:
    """A fixation event.

    Keyword arguments:
    time (integer) -- the timestamp of the starting sample.
    x, y (float)   -- the coordinates of the starting sample.

    Public methods:
    still_going(dict) -- decide whether a sample continues this fixation event.
    finish() -- calculate a bunch of properties when the fixation is done.

    as_dict() -- Dump the data as a dictionary:
        'trial' (int): number of trial
        'ordinal' (int): ordinal number of fixation within the trial
        'start' (int): timestamp of start
        'end' (int): timestamp of end
        'duration' (int): duration in ms
        'avgx' (float): average x for all samples within fixation
        'avgy' (float): average y
        'quadrant' (int): the quadrant of the screen
    """

    def __init__(self, sample, threshold=50, min_duration=100):
        self.start_time = sample['time']
        self.samples = [sample]
        # the dispersion window starts as a point
        self.max_x, self.max_y = sample['avgx'], sample['avgy']
        self.min_x, self.min_y = self.max_x, self.max_y
        # the dispersion threshold and required duration
        self.threshold_squared = threshold**2
        self.min_duration = min_duration

    def still_going(self, sample):
        """Decides whether the sample continues this fixation."""

        diameter_squared = (self.max_x - self.min_x)**2
        + (self.max_y - self.min_y)**2
        # compare the diameter of our dispersion window to the threshold
        # but don't bother taking the square root
        # so we use the square of the threshold
        if diameter_squared < self.threshold_squared:
            return True
        else:
            return False

    def finish(self):
        pass

    def as_dict(self):
        pass

if __name__ == '__main__':
    with open('../test-data/subject-998.tsv') as csvfile:
        reader = csv.DictReader(csvfile, dialect='excel-tab')

        # get to the first sample data
        row = {'timestamp': 'MSG'}
        while row['timestamp'] == 'MSG':
            row = reader.next()
            continue

        fixations = []
        current_fix = Fixation(row)

        for row in reader:
            if row['timestamp'] == 'MSG':
                # this isn't sample data, so ignore
                continue
            elif current_fix.still_going(row):
                current_fix.samples.append(row)
            else:
                current_fix.finish()
                if current_fix.duration >= current_fix.min_duration:
                    fixations.append(current_fix)
                # if it's too short, it gets overwritten
                current_fix = Fixation(row)
