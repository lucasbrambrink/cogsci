from django.db import models
import numpy as np


class Participant(models.Model):
    name = models.CharField(max_length=255)
    is_control = models.BooleanField(default=False)
    agg_transfer_score = models.IntegerField(default=0)

    def set_transfer_score(self):
        self.agg_transfer_score = sum(self.transfertest_set
                                          .filter(is_active=True)
                                          .values_list('score', flat=True))
        self.save(update_fields=['agg_transfer_score'])

    def assign_condition(self):
        tc = self.testconceptanswer_set.first()
        self.is_control = tc.is_control
        self.save(update_fields=['is_control'])

    def __repr__(self):
        return self.name


class ActiveManager(models.Manager):

    def get_queryset(self, *args, **kwargs):
        return super(ActiveManager, self) \
            .get_queryset(*args, **kwargs) \
            .filter(is_active=True)


class TestConcept(models.Model):
    name = models.CharField(max_length=255)
    compressed_name = models.CharField(max_length=255, default='')
    too_easy = models.BooleanField(default=False)
    too_hard = models.BooleanField(default=False)
    percentage_score = models.DecimalField(max_digits=12, decimal_places=5, default=0)
    is_active = models.BooleanField(default=True)

    published = ActiveManager()
    objects = models.Manager()

    def assign_compressed(self):
        self.compressed_name = self.name.replace(' ', '')
        self.save(update_fields=['compressed_name'])

    def __repr__(self):
        return '{}: {}'.format(self.name, self.percentage_score)

    def __str__(self):
        return self.__repr__()


    def assign_boundary_metrics(self):
        """too easy or too hard"""
        THRESHOLD = 0.10
        low = TransferTest.published \
            .filter(concept__compressed_name=self.compressed_name) \
            .values_list('score', flat=True)
        max_score = sum(3 for x in low)
        avg = sum(low) / max_score

        print(sum(low), max_score, avg, self.name)
        self.percentage_score = avg
        if avg <= THRESHOLD:
            self.too_hard = True
        elif avg >= max_score * (1 - THRESHOLD):
            self.too_easy = True
        self.save()


class TestConceptAnswer(models.Model):
    EXAM = 'Problem Oriented - Training'
    CONTROL = 'Memory - Training'

    test_concept = models.ForeignKey(to='TestConcept')
    solution_reached = models.NullBooleanField(null=True)
    solution_strategy = models.CharField(max_length=255, default='')
    confidence_level = models.IntegerField(null=True)
    solution_score = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    familiarity1 = models.IntegerField()
    familiarity2 = models.IntegerField()
    familiarity3 = models.IntegerField()
    aha_during_explanation = models.BooleanField()
    surprise_rating = models.IntegerField(null=True)
    is_control = models.BooleanField()
    participant = models.ForeignKey(to="Participant")
    is_active = models.BooleanField(default=True)

    published = ActiveManager()
    objects = models.Manager()


    def __repr__(self):
        return '{}: {}'.format(self.test_concept.name, str(self.participant))

    @classmethod
    def fields_in_order(cls):
        return (x for x in ('solution_reached', 'solution_strategy', 'confidence_level',
                            'IGNORE', 'solution_score', 'familiarity1', 'familiarity2', 'familiarity3',
                            'aha_during_explanation', 'surprise_rating'))

    @classmethod
    def control_fields(cls):
        return (x for x in ('familiarity1', 'familiarity2', 'familiarity3', 'IGNORE', 'aha_during_explanation'))

    def update_is_active(self):
        self.is_active = self.test_concept.is_active
        self.save()


class TransferTest(models.Model):
    participant = models.ForeignKey(to="Participant")
    test_concept_answer = models.ForeignKey(to='TestConceptAnswer')
    concept = models.ForeignKey(to='TestConcept', null=True)
    name = models.CharField(max_length=255)
    score = models.DecimalField(max_digits=12, decimal_places=2)
    aha_during_transfer = models.NullBooleanField(null=True)
    notes = models.CharField(max_length=255, default='', null=True)
    is_active = models.BooleanField(default=True)

    published = ActiveManager()
    objects = models.Manager()

    def update_is_active(self):
        self.is_active = self.concept.is_active
        self.save()

    def update_score(self):
        if self.score == 0.50:
            self.score = 1
        elif self.score == 0.75:
            self.score = 2
        elif self.score == 1:
            self.score = 3
        self.save(update_fields=['score'])

    def set_test_concept(self):
        self.concept = self.test_concept_answer.test_concept
        self.save(update_fields=['concept'])


class Analysis(models.Model):
    EASINESS_THRESHOLD = 0.1
    name = models.CharField(max_length=255)

    @classmethod
    def graph(cls):
        """compare control vs exp for transfer scores"""
        import matplotlib.pyplot as plt
        from .utils import Stat
        exp = Participant.objects.filter(is_control=False).values_list('agg_transfer_score', flat=True)
        control = Participant.objects.filter(is_control=True).values_list('agg_transfer_score', flat=True)
        s = Stat(exp)
        c = Stat(control)

        print("Experiment: ", s.mean, s.std)
        print("Control: ", c.mean, c.std)

        # plt.bar(xs, ys_control, color='r')
        # plt.bar(xs, ys_experiment, color='b')
        # plt.show())
        plt.boxplot([exp, control],labels=['Experiment', 'Control'])
        plt.title('Distribution of aggregate transfer scores by participant')
        plt.ylabel('Agg. Score')
        plt.show()

    @classmethod
    def more_graphing(cls):
        """if solution stratgey I, A, 0 , average transfer score"""
        import matplotlib.pyplot as plt
        from .utils import Stat
        exp = list(map(float, TransferTest.published\
            .filter(test_concept_answer__solution_strategy='I',
                    test_concept_answer__is_control=False)\
            .values_list('score', flat=True)))
        control = list(map(float, TransferTest.published \
            .filter(test_concept_answer__solution_strategy='A',
                    test_concept_answer__is_control=False) \
            .values_list('score', flat=True)))
        nulls = list(map(float, TransferTest.published \
                           .filter(test_concept_answer__solution_strategy='0',
                                   test_concept_answer__is_control=False) \
                           .values_list('score', flat=True)))
        s = Stat(exp)
        c = Stat(control)
        n = Stat(nulls)
        plt.boxplot([exp, control, nulls], labels=['I', 'A', '0'])

        print("I: ", s.mean, s.std, len(s.data))
        print("A: ", c.mean, c.std, len(c.data))
        print("Null", n.mean, n.std, len(n.data))
        # plt.title('Distribution of aggregate transfer scores by participant')
        # plt.ylabel('Agg. Score')
        lines = [
            'I: ' + ','.join(map(str, exp)),
            'A: ' + ','.join(map(str, control)),
            '0: ' + ','.join(map(str, nulls)),
            'Aha during Transfer Exp True ' + ','.join(map(str, TransferTest.published.filter(participant__is_control=False,
                                                                               aha_during_transfer=True)
                                                 .values_list('score', flat=True))),
            'Aha during Transfer  Exp False ' + ','.join(map(str, TransferTest.published.filter(participant__is_control=False,
                                                                               aha_during_transfer=False).values_list(
                'score', flat=True))),
            'Aha during Transfer  Control True ' + ','.join(map(str, TransferTest.published.filter(participant__is_control=True,
                                                                               aha_during_transfer=True).values_list(
                'score', flat=True))),
            'Aha during Transfer  Control False ' + ','.join(map(str, TransferTest.published.filter(participant__is_control=True,
                                                                               aha_during_transfer=False).values_list(
                'score', flat=True))),
            'Aha during Explanation Exp True ' + ','.join(map(str, TransferTest.published.filter(participant__is_control=False,
                                                                                          test_concept_answer__aha_during_explanation=True)
                                                       .values_list('score', flat=True))),
            'Aha during Explanation Exp False ' + ','.join(
                map(str, TransferTest.published.filter(participant__is_control=False,
                                                       test_concept_answer__aha_during_explanation=False).values_list(
                    'score', flat=True))),
            'Aha during Explanation Control True ' + ','.join(
                map(str, TransferTest.published.filter(participant__is_control=True,
                                                       test_concept_answer__aha_during_explanation=True).values_list(
                    'score', flat=True))),
            'Aha during Explanation Control False ' + ','.join(
                map(str, TransferTest.published.filter(participant__is_control=True,
                                                       test_concept_answer__aha_during_explanation=False).values_list(
                    'score', flat=True))),
        ]

        with open('transfer_scores_data_set.csv', 'w') as export_file:
            for line in lines:
                export_file.write(line + '\n')

        plt.show()

    @classmethod
    def aha_during_explanation(cls):
        import matplotlib.pyplot as plt
        from .utils import Stat
        exp = list(map(lambda x: float(x * 1000), TransferTest.published \
                       .filter(test_concept_answer__aha_during_explanation=True,
                               test_concept_answer__is_control=False) \
                       .values_list('score', flat=True)))
        control = list(map(lambda x: float(x * 1000), TransferTest.published \
                           .filter(test_concept_answer__aha_during_explanation=True,
                                   test_concept_answer__is_control=True) \
                           .values_list('score', flat=True)))
        s = Stat(exp)
        c = Stat(control)

        print("True Aha Exp: ", s.mean, s.std, len(s.data))
        print("True Aha Cont: ", c.mean, c.std, len(c.data))

        import matplotlib.pyplot as plt
        from .utils import Stat
        exp1 = list(map(lambda x: float(x * 1000), TransferTest.published \
                       .filter(test_concept_answer__aha_during_explanation=False,
                               test_concept_answer__is_control=False) \
                       .values_list('score', flat=True)))
        control1 = list(map(lambda x: float(x * 1000), TransferTest.published \
                           .filter(test_concept_answer__aha_during_explanation=False,
                                   test_concept_answer__is_control=True) \
                           .values_list('score', flat=True)))
        s1 = Stat(exp1)
        c1 = Stat(control1)

        print("False Aha Exp: ", s1.mean, s1.std, len(s1.data))
        print("False Aha Cont: ", c1.mean, c1.std, len(c1.data))
        # plt.title('Distribution of aggregate transfer scores by participant')
        # plt.ylabel('Agg. Score')
        plt.boxplot([exp, control, exp1, control1], labels=['Aha true, Exp', 'Aha true, Cont', 'Aha false, Exp', "aha false, Cont"])

        plt.show()

    @classmethod
    def aha_during_transfer(cls):
        import matplotlib.pyplot as plt
        from .utils import Stat
        exp = list(map(lambda x: float(x), TransferTest.published \
                       .filter(aha_during_transfer=True,
                               test_concept_answer__is_control=False) \
                       .values_list('score', flat=True)))
        control = list(map(lambda x: float(x), TransferTest.published \
                           .filter(aha_during_transfer=True,
                                   test_concept_answer__is_control=True) \
                           .values_list('score', flat=True)))

        exp1 = list(map(lambda x: float(x), TransferTest.published \
                        .filter(aha_during_transfer=False,
                                test_concept_answer__is_control=False) \
                        .values_list('score', flat=True)))
        control1 = list(map(lambda x: float(x), TransferTest.published \
                            .filter(aha_during_transfer=False,
                                    test_concept_answer__is_control=True) \
                            .values_list('score', flat=True)))

        nulls_exp = list(map(lambda x: float(x), TransferTest.published \
                        .filter(aha_during_transfer=None,
                                test_concept_answer__is_control=False) \
                        .values_list('score', flat=True)))
        nulls_cont = list(map(lambda x: float(x), TransferTest.published \
                            .filter(aha_during_transfer=None,
                                    test_concept_answer__is_control=True) \
                            .values_list('score', flat=True)))

        s = Stat(exp)
        c = Stat(control)
        s1 = Stat(exp1)
        c1 = Stat(control1)
        s2 = Stat(nulls_exp)
        c2 = Stat(nulls_cont)

        print("True Aha Exp: ", s.mean, s.std, len(s.data))
        print("True Aha Cont: ", c.mean, c.std, len(c.data))

        print("False Aha Exp: ", s1.mean, s1.std, len(s1.data))
        print("False Aha Cont: ", c1.mean, c1.std, len(c1.data))

        print("None Aha Exp: ", s2.mean, s2.std, len(s2.data))
        print("None Aha Cont: ", c2.mean, c2.std, len(c2.data))
        plt.boxplot([exp, control, exp1, control1, nulls_exp, nulls_cont])

        plt.show()
    # @classmethod
    # def graph_test_condition_scores(cls):
    #     import matplotlib.pyplot as plt
    #     test_cases_control = list(TestConceptAnswer.published.filter(is_control=True).values_list('pk', flat=True))
    #     test_cases_exp = list(TestConceptAnswer.published.filter(is_control=False).values_list('pk', flat=True))
    #     indexes = range(len(test_cases_control))
    #     print(test_cases_control)
    #     ys_c = []
    #     ys_exp = []
    #     for i in indexes:
    #         ys_c.append(sum(TransferTest.published\
    #             .filter(concept_id=test_cases_control[i])\
    #             .values_list('score', flat=True)))
    #         ys_exp.append(sum(TransferTest.published \
    #                     .filter(concept_id=test_cases_exp[i]) \
    #                     .values_list('score', flat=True)))
    #
    #     plt.scatter(indexes, ys_exp, c='r')
    #     # plt.scatter(indexes, ys_c, c='b')
    #     # plt.plot(xs, ys, 'r--')
    #     # plt.plot(xs, yys, 'b--')
    #     plt.show()

    @classmethod
    def stat_analysis(cls):
        """compare control vs exp for transfer scores"""
        import matplotlib.pyplot as plt
        from .utils import Stat

        xs = [0, 1, 2, 3]

        exp = list(map(float, TransferTest.published.filter(participant__is_control=False)\
            .values_list('score', flat=True)))
        control = list(map(float, TransferTest.published.filter(participant__is_control=True)\
            .values_list('score', flat=True)))
        s_experiment = Stat(exp)
        s_control = Stat(control)
        print('Mean', s_experiment.mean, s_control.mean)
        print('Std', s_experiment.std, s_control.std)

        exp = []
        for p in Participant.objects.filter(is_control=False):
            exp.append(sum(map(float, TransferTest.published.filter(participant=p).values_list('score', flat=True))))
        control = []
        for p in Participant.objects.filter(is_control=True):
            control.append(sum(map(float, TransferTest.published.filter(participant=p).values_list('score', flat=True))))

        s_experiment = Stat(exp)
        s_control = Stat(control)
        print('Agg. Mean', s_experiment.mean, s_control.mean)
        print('Agg. Std', s_experiment.std, s_control.std)

    @classmethod
    def aha_analysis(cls):
        """compare control vs exp for transfer scores"""
        import matplotlib.pyplot as plt
        from .utils import Stat

        t = list(map(float,
                    TransferTest.published.filter(participant__is_control=False,
                                                  test_concept__aha_during_explanation=True).values_list('score', flat=True)))
        transfer_score_yes_aha_experiment = sum(t) / len(t)
        s = len(t)
        t = list(map(float,
                TransferTest.published.filter(participant__is_control=False,
                                            test_concept__aha_during_explanation=False).values_list('score', flat=True)))
        q = len(t)
        transfer_score_no_aha_experiment = sum(t) / len(t)
        print("Experiment: Yes aha", transfer_score_yes_aha_experiment, s)
        print("Experiment: No aha", transfer_score_no_aha_experiment, q)

        t = list(map(float,
                    TransferTest.published.filter(participant__is_control=True,
                                                test_concept__aha_during_explanation=True).values_list('score',
                                                                                                       flat=True)))
        s = len(t)
        transfer_score_yes_aha_experiment = sum(t) / len(t)
        t = list(map(float,
                    TransferTest.published.filter(participant__is_control=True,
                                                test_concept__aha_during_explanation=False).values_list('score',
                                                                                                        flat=True)))
        q = len(t)
        transfer_score_no_aha_experiment = sum(t) / len(t)
        print("Control: Yes aha", transfer_score_yes_aha_experiment, s)
        print("Control: No aha", transfer_score_no_aha_experiment, q)

        t = list(map(float,
                    TransferTest.published.filter(participant__is_control=False,
                                                  test_concept__solution_strategy='I').values_list('score',
                                                                                                       flat=True)))
        s = len(t)
        transfer_score_yes_aha_experiment = sum(t) / len(t)
        t= list(map(float,
                    TransferTest.published.filter(participant__is_control=False,
                                                  test_concept__solution_strategy='A').values_list('score',
                                                                                                        flat=True)))
        q = len(t)
        transfer_score_no_aha_experiment = sum(t) / len(t)
        print("Solution strategy I:", transfer_score_yes_aha_experiment, s)
        print("Solution strategy A:", transfer_score_no_aha_experiment, q)
        # transfer_score_control = \
        #     sum(map(float,
        #             TransferTest.objects.filter(participant__is_control=True,
        #                                         test_concept__aha_during_explanation=True)))
        #
        # exp = list(map(float, TransferTest.objects.filter(participant__is_control=False) \
        #                .values_list('score', flat=True)))
        # control = list(map(float, TransferTest.objects.filter(participant__is_control=True) \
        #                    .values_list('score', flat=True)))
        # s_experiment = Stat(exp)
        # s_control = Stat(control)
        # print('Mean', s_experiment.mean, s_control.mean)
        # print('Std', s_experiment.std, s_control.std)
        #
        # exp = []
        # for p in Participant.objects.filter(is_control=False):
        #     exp.append(sum(map(float, TransferTest.objects.filter(participant=p).values_list('score', flat=True))))
        # control = []
        # for p in Participant.objects.filter(is_control=True):
        #     control.append(
        #         sum(map(float, TransferTest.objects.filter(participant=p).values_list('score', flat=True))))
        #
        # s_experiment = Stat(exp)
        # s_control = Stat(control)
        # print('Agg. Mean', s_experiment.mean, s_control.mean)
        # print('Agg. Std', s_experiment.std, s_control.std)


    @classmethod
    def stat_analysis_by_aha(cls):
        """compare control vs exp for transfer scores"""
        import matplotlib.pyplot as plt
        from .utils import Stat

        exp = []
        for p in Participant.objects.filter(is_control=False):
            exp.append(TransferTest.published.filter(participant=p, aha_during_transfer=True).count())
        control = []
        for p in Participant.objects.filter(is_control=True):
            control.append(TransferTest.published.filter(participant=p, aha_during_transfer=True).count())

        s_experiment = Stat(exp)
        s_control = Stat(control)
        print("AHA == True")
        print('Agg. Mean', s_experiment.mean, s_control.mean)
        print('Agg. Std', s_experiment.std, s_control.std)
        print("Total count", sum(exp), sum(control))
        exp = []
        for p in Participant.objects.filter(is_control=False):
            exp.append(TransferTest.published.filter(participant=p, aha_during_transfer=False).count())
        control = []
        for p in Participant.objects.filter(is_control=True):
            control.append(TransferTest.published.filter(participant=p, aha_during_transfer=False).count())

        s_experiment = Stat(exp)
        s_control = Stat(control)
        print("AHA == False")
        print('Agg. Mean', s_experiment.mean, s_control.mean)
        print('Agg. Std', s_experiment.std, s_control.std)
        print("Total count", sum(exp), sum(control))



    @classmethod
    def agg_transfer_score(cls):
        from .utils import Stat
        exp = Participant.objects.filter(is_control=False).values_list('agg_transfer_score', flat=True)
        control = Participant.objects.filter(is_control=True).values_list('agg_transfer_score', flat=True)
        s_experiment = Stat(exp)
        s_control = Stat(control)
        print('Mean', s_experiment.mean, s_control.mean)
        print('Std', s_experiment.std, s_control.std)
        print("Total count", sum(exp), sum(control))

    @classmethod
    def compared_insights(cls):
        """per person, # aha explan. agg. transfer score
        """
        import matplotlib.pyplot as plt
        from .utils import Stat
        xs, ys, yys, xss = [], [], [], []
        for p in Participant.objects.filter(is_control=False):
            xs.append(p.agg_transfer_score)
            ys.append(p.testconceptanswer_set.filter(aha_during_explanation=True).count())
        for p in Participant.objects.filter(is_control=True):
            xss.append(p.agg_transfer_score)
            yys.append(p.testconceptanswer_set.filter(aha_during_explanation=True).count())
            # yyys.append(p.testconceptanswer_set.filter(solution_strategy='0').count())


        plt.plot(np.unique(xs), np.poly1d(np.polyfit(xs, ys, 1))(np.unique(xs)), color='r')
        plt.plot(np.unique(xss), np.poly1d(np.polyfit(xss, yys, 1))(np.unique(xss)), color='b')
        # plt.plot(np.unique(xs), np.poly1d(np.polyfit(xs, yyys, 1))(np.unique(xs)), color='g')
        plt.scatter(xs, ys, color='r')
        plt.scatter(xss, yys, color='b')
        # plt.scatter(xs, yyys, color='g')
        plt.title('agg. transfer score vs # Aha-moments during explanation')
        plt.ylabel('# Aha Moments')
        plt.xlabel('Agg. transfer score')
        plt.legend(['Experiment', 'Control'])
        plt.show()

    @classmethod
    def aggregate_aha_during_explanation(cls):
        """per person, # aha explan. agg. transfer score
        """
        import matplotlib.pyplot as plt
        from .utils import Stat
        xs, ys, yys, xss = [], [], [], []
        for p in Participant.objects.filter(is_control=False):
            ys.append(p.testconceptanswer_set.filter(aha_during_explanation=True).count())
        for p in Participant.objects.filter(is_control=True):
            yys.append(p.testconceptanswer_set.filter(aha_during_explanation=True).count())
            # yyys.append(p.testconceptanswer_set.filter(solution_strategy='0').count())



        plt.boxplot([ys, yys],labels=['Experiment', 'Control'])
        plt.title('Distribution of # aha moments during explanation by participant')
        plt.ylabel('# Aha moments')
        plt.show()

    @classmethod
    def print_scores(cls):
        """"""
        lines = []
        headers = (
            'Name',
            'Agg. Transfer score',
            'solution strat I',
            'solution strat A',
            'solution strat 0',
            'aha explanation Y',
            'aha explanation N',
            'aha transfer Y',
            'aha transfer N',
            'aha transfer 0'
        )
        lines.append(headers)
        for p in Participant.objects.all():
            lines.append((
                p.name,
                p.agg_transfer_score,
                p.testconceptanswer_set.filter(solution_strategy='I').count(),
                p.testconceptanswer_set.filter(solution_strategy='A').count(),
                p.testconceptanswer_set.filter(solution_strategy='0').count(),
                p.testconceptanswer_set.filter(aha_during_explanation=True).count(),
                p.testconceptanswer_set.filter(aha_during_explanation=False).count(),
                p.transfertest_set.filter(aha_during_transfer=True).count(),
                p.transfertest_set.filter(aha_during_transfer=False).count(),
                p.transfertest_set.filter(aha_during_transfer=None).count(),
            ))

        with open('export.csv', 'w') as export_file:
            for line in lines:
                export_file.write(','.join(map(str, line)) + '\n')


    @classmethod
    def compared_surprise(cls):
        """per person, # aha explan. agg. transfer score
        """
        import matplotlib.pyplot as plt
        from .utils import Stat
        xs, ys = [], []
        values = []
        for score in range(1, 6):
            xs.append(score)
            qs = TransferTest.objects.filter(participant__is_control=False,
                                             test_concept_answer__surprise_rating=score)\
                            .values_list('score', flat=True)
            values.append(list(map(float, qs)))
            t = Stat(list(map(float, qs)))
            ys.append(t.mean)

        # xss, yys = [], []
        # for t in TransferTest.objects.filter(participant__is_control=False):
        #     yys.append(float(t.score))
        #     xss.append(float(t.test_concept_answer.surprise_rating))

        # plt.boxplot(values, labels=['1', '2', '3', '4', '5'])

            # for p in Participant.objects.filter(is_control=False):
            # xs.append(p.agg_transfer_score)
        #     ys.append(p.testconceptanswer_set.filter(aha_during_explanation=True).count())
        # for p in Participant.objects.filter(is_control=True):
        #     xss.append(p.agg_transfer_score)
        #     yys.append(p.testconceptanswer_set.filter(aha_during_explanation=True).count())
            # yyys.append(p.testconceptanswer_set.filter(solution_strategy='0').count())

        lines = []
        headers = (
            'Surprise rating',
            'Transfer score',
        )
        lines.append(headers)
        for x, value_list in zip(xs, values):
            for value in value_list:
                lines.append((x, value))

        with open('surprise_rating.csv', 'w') as export_file:
            for line in lines:
                export_file.write(','.join(map(str, line)) + '\n')
        plt.plot(np.unique(xs), np.poly1d(np.polyfit(xs, ys, 2))(np.unique(xs)), color='r')
        # plt.plot(np.unique(xss), np.poly1d(np.polyfit(xss, yys, 2))(np.unique(xss)), color='b')
        # plt.plot(np.unique(xs), np.poly1d(np.polyfit(xs, yyys, 1))(np.unique(xs)), color='g')
        plt.scatter(xs, ys, color='b')
        # plt.scatter(xss, yys, color='b')
        # plt.scatter(xs, yyys, color='g')
        plt.title('Surprise rating vs average transfer score')
        plt.ylabel('Average transfer score')
        plt.xlabel('Surprise rating')
        # plt.legend(['Experiment', 'Control'])
        plt.show()


    @classmethod
    def other_values(cls):
        """"""
        lines = []
        for cond in (True, False):
            lines.append((
                ['Aha expl. FALSE' if not cond else 'Aha expl. TRUE'] + list(TestConceptAnswer.objects.filter(is_control=False)
                    .filter(aha_during_explanation=cond).values_list('confidence_level', flat=True)),
                # list(TestConceptAnswer.objects.filter(is_control=False)
                #     .filter(aha_during_explanation=False).values_list('confidence_level', flat=True)),
                # list(t.testconceptanswer_set.filter(is_control=True)
                #     .filter(aha_during_explanation=True).values_list('confidence_level', flat=True)),
                # list(t.testconceptanswer_set.filter(is_control=True)
                #     .filter(aha_during_explanation=False).values_list('confidence_level', flat=True)),
            ))

        with open('confidence_level.csv', 'w') as export_file:
            for line in lines:
                export_file.write(','.join(map(str, line)) + '\n')

