import unittest

import numpy as np

from abstention.label_shift import EMImbalanceAdapter


class TestEMImbalanceAdapter(unittest.TestCase):

    def setUp(self):
        self.valid_posterior_probs = np.array([
            [0.70, 0.20, 0.10],
            [0.20, 0.60, 0.20],
            [0.10, 0.20, 0.70],
        ])
        self.tofit_posterior_probs = np.array([
            [0.20, 0.75, 0.05],
            [0.20, 0.65, 0.15],
            [0.10, 0.80, 0.10],
        ])

    def test_support_masks_target_prior_and_adapted_probabilities(self):
        adapter = EMImbalanceAdapter()
        support = np.array([1, 0, 1])
        fitted = adapter(
            tofit_initial_posterior_probs=self.tofit_posterior_probs,
            valid_posterior_probs=self.valid_posterior_probs,
            class_label_support=support)

        target_prior = fitted.multipliers * np.mean(
            self.valid_posterior_probs, axis=0)
        adapted = fitted(self.tofit_posterior_probs)

        np.testing.assert_allclose(target_prior.sum(), 1.0)
        self.assertTrue(np.all(target_prior >= 0))
        self.assertEqual(target_prior[1], 0.0)
        np.testing.assert_allclose(adapted[:, 1], 0.0)
        np.testing.assert_allclose(adapted.sum(axis=1), 1.0)

    def test_boolean_and_integer_masks_are_equivalent(self):
        adapter = EMImbalanceAdapter()
        boolean_fit = adapter(
            self.tofit_posterior_probs, self.valid_posterior_probs,
            class_label_support=np.array([True, False, True]))
        integer_fit = adapter(
            self.tofit_posterior_probs, self.valid_posterior_probs,
            class_label_support=np.array([1, 0, 1]))

        np.testing.assert_allclose(
            boolean_fit.multipliers, integer_fit.multipliers)

    def test_no_support_preserves_existing_behavior(self):
        adapter = EMImbalanceAdapter()
        default_fit = adapter(
            self.tofit_posterior_probs, self.valid_posterior_probs)
        explicit_none_fit = adapter(
            self.tofit_posterior_probs, self.valid_posterior_probs,
            class_label_support=None)

        np.testing.assert_allclose(
            default_fit.multipliers, explicit_none_fit.multipliers)

    def test_invalid_support_is_rejected(self):
        adapter = EMImbalanceAdapter()
        invalid_masks = [
            np.array([1, 0]),
            np.array([1, 2, 0]),
            np.array([0, 0, 0]),
            np.array([[1, 0, 1]]),
        ]
        for mask in invalid_masks:
            with self.subTest(mask=mask):
                with self.assertRaises(ValueError):
                    adapter(self.tofit_posterior_probs,
                            self.valid_posterior_probs,
                            class_label_support=mask)

    def test_supported_class_with_zero_validation_prior_is_rejected(self):
        adapter = EMImbalanceAdapter()
        valid_posterior_probs = np.array([
            [0.7, 0.3, 0.0],
            [0.4, 0.6, 0.0],
        ])
        with self.assertRaises(ValueError):
            adapter(self.tofit_posterior_probs,
                    valid_posterior_probs,
                    class_label_support=np.array([1, 0, 1]))

    def test_support_works_with_binary_probability_inputs(self):
        adapter = EMImbalanceAdapter()
        fitted = adapter(
            tofit_initial_posterior_probs=np.array([[0.8], [0.3], [0.6]]),
            valid_posterior_probs=np.array([[0.4], [0.6], [0.5]]),
            class_label_support=np.array([0, 1]))

        adapted = fitted(np.array([[0.8], [0.3], [0.6]]))
        np.testing.assert_allclose(adapted, np.ones((3, 1)))
