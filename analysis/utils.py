import pandas as pd
import numpy as np
from numpy import float as np_float
import os
from .models import TestConcept, TestConceptAnswer, Participant, TransferTest


class ParseData(object):
    def __init__(self, file_path):
        assert os.path.isfile(file_path)
        self.path_name = file_path
        self.xl = pd.ExcelFile(file_path)
        self.exp = self.parse_sheet(self.xl.sheet_names[0])
        self.control = self.parse_sheet(self.xl.sheet_names[1])
        self.transfer = self.parse_sheet(self.xl.sheet_names[2])

    def __call__(self, *args, **kwargs):
        self.parse_information(self.exp)
        self.parse_information_control(self.control)
        for t in TestConcept.objects.all():
            t.assign_compressed()

        self.parse_transfer(self.transfer)
        for t in TransferTest.objects.all():
            t.update_score()
            t.set_test_concept()

        for p in Participant.objects.all():
            p.assign_condition()
            p.set_transfer_score()

        t = TestConcept.objects.get(name__icontains='self mimicry')
        t.is_active = False
        t.save()

        for t in TransferTest.objects.all():
            t.update_is_active()

        for t in TestConceptAnswer.objects.all():
            t.update_is_active()


    def delete_all(self):
        TestConcept.objects.all().delete()
        Participant.objects.all().delete()
        TransferTest.objects.all().delete()
        TestConceptAnswer.objects.all().delete()

    def parse_sheet(self, sheet_name):
        df = self.xl.parse(sheet_name)
        return df

    def parse_information(self, df):
        for row in df.index[1:]:
            participant = Participant()
            test_case = None
            fields = TestConceptAnswer.fields_in_order()
            for column in df.columns:
                if column == 'Name':
                    participant.name = df.get_value(row, column)
                    participant.save()
                    continue

                if 'Unnamed' not in column:
                    if test_case is not None:
                        participant.save()
                        test_case.participant = participant
                        print(test_case.__dict__)
                        test_case.save()
                        fields = TestConceptAnswer.fields_in_order()
                    try:
                        name = TestConcept.objects.get(name=column)
                    except TestConcept.DoesNotExist:
                        name = TestConcept(name=column)
                        name.save()
                    test_case = TestConceptAnswer(
                        is_control=False,
                        test_concept=name)

                field = next(fields)
                if field is 'IGNORE':
                    continue


                value = df.get_value(row, column)
                if value in ('N', 'n'):
                    value = False
                elif value in ('Y', 'y'):
                    value = True
                elif type(value) is np_float:
                    value = 0

                # elif type(value) is str and value.lower() == 'nan':
                #     value = 0
                # print(value, type(value), type(value) is np_float)
                setattr(test_case, field, value)

            participant.save()
            test_case.participant = participant
            print(test_case.__dict__)
            test_case.save()

    def parse_information_control(self, df):
        for row in df.index[1:]:
            participant = Participant()
            test_case = None
            fields = TestConceptAnswer.control_fields()
            for column in df.columns:
                if column == 'Name':
                    participant.name = df.get_value(row, column)
                    participant.save()
                    continue

                if 'Unnamed' not in column:
                    if test_case is not None:
                        participant.save()
                        test_case.participant = participant
                        print(test_case.__dict__)
                        test_case.save()
                        fields = TestConceptAnswer.control_fields()
                    try:
                        name = TestConcept.objects.get(name=column)
                    except TestConcept.DoesNotExist:
                        name = TestConcept(name=column)
                        name.save()
                    test_case = TestConceptAnswer(
                        is_control=True,
                        test_concept=name)

                field = next(fields)
                if field is 'IGNORE':
                    continue


                value = df.get_value(row, column)
                if value in ('N', 'n'):
                    value = False
                elif value in ('Y', 'y'):
                    value = True
                elif type(value) is np_float:
                    value = 0

                # elif type(value) is str and value.lower() == 'nan':
                #     value = 0
                # print(value, type(value), type(value) is np_float)
                setattr(test_case, field, value)

            participant.save()
            test_case.participant = participant
            print(test_case.__dict__)
            test_case.save()

    def parse_transfer(self, df):
        for row in df.index:
            count = 0
            transfer = None
            for column in df.columns:
                if column == 'Name':
                    print('Name', df.get_value(row, column))
                    participant = Participant.objects.get(name__icontains=df.get_value(row, column))
                    continue
                elif column == 'Condition':
                    continue

                value = df.get_value(row, column)
                if value in ('N', 'n'):
                    value = False
                elif value in ('Y', 'y', 'U'):
                    value = True
                elif value in ("N/A", 'n/a', 'nan', 'A') or type(value) is np_float:
                    value = None
                elif value in (0, '0') and count == 2:
                    value = None
                if count == 0:
                    if transfer is not None:
                        transfer.participant = participant
                        name = getattr(transfer, 'hold_name').replace('Transfer_', '').replace(' ', '')
                        if name == "ExpolitativeCompetition":
                            name = "ExploitativeCompetition"
                        print(participant, name)
                        transfer.test_concept_answer = TestConceptAnswer.objects.get(
                            participant=participant,
                            test_concept__compressed_name=name
                        )
                        print(transfer.__dict__)
                        transfer.save()
                    transfer = TransferTest(
                        name=column,
                        notes=value
                    )
                    count += 1
                elif count == 1:
                    setattr(transfer, 'hold_name', column)
                    transfer.score = float(value)
                    count += 1
                elif count == 2:
                    transfer.aha_during_transfer = value
                    count = 0

            transfer.participant = participant
            name = getattr(transfer, 'hold_name').replace('Transfer_', '').replace(' ', '')
            if name == "ExpolitativeCompetition":
                name = "ExploitativeCompetition"
            print(participant, name)
            transfer.test_concept_answer = TestConceptAnswer.objects.get(
                participant=participant,
                test_concept__compressed_name=name
            )
            print(transfer.__dict__)
            transfer.save()


class Stat(object):

    def __init__(self, data):
        self.data = data

    @property
    def mean(self):
        if not len(self.data):
            return 0
        return sum(self.data) / len(self.data)

    @property
    def sqr_dev(self):
        return map(lambda x: x ** 2, self.deviations)

    @property
    def deviations(self):
        mean = self.mean
        return [d - mean for d in self.data]

    @property
    def var(self):
        if not len(list(self.sqr_dev)):
            return 0
        return sum(self.sqr_dev) / len(list(self.sqr_dev))

    @property
    def std(self):
        return self.var ** 0.5


class StatisticalSignificance(object):
    """calculate p-value"""
    SIGNIFICANCE_LEVEL = 0.05




    H0_HYPOTHESIS = 'No relationship can be found'


