"""Student noise margin for later arms (PI 2026-11-17): the larger of the sham-pair and naive-pair margins."""
from zoo.assay_rule import seed_status
from zoo.phase4b_report import student_margin


def test_student_margin_is_the_larger_of_the_two():
    assert student_margin(0.0, 0.311) == 0.311
    assert student_margin(0.422, 0.311) == 0.422


def test_zero_sham_margin_no_longer_makes_a_one_seed_difference_count():
    score = {"L": -0.26, "lower_bound": False}       # seed 4, student 1x against its sham student
    assert seed_status(score, 0.0) == "faster"
    assert seed_status(score, student_margin(0.0, 0.311)) == "inside"
