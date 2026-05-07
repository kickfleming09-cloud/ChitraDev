from __future__ import annotations

import unittest

from app.prompting import (
    FacePrompt,
    compose_generation_prompt,
    compose_negative_prompt,
    compose_prompt_warning,
    derive_negative_prompt_terms,
)


class PromptingTests(unittest.TestCase):
    def test_compose_generation_prompt_for_prompt_only(self) -> None:
        prompt = compose_generation_prompt(
            scene_prompt="a futuristic mountain observatory at sunrise",
            face_prompts=[],
        )

        self.assertEqual(prompt, "a futuristic mountain observatory at sunrise.")

    def test_compose_generation_prompt_for_single_face(self) -> None:
        prompt = compose_generation_prompt(
            scene_prompt="cinematic portrait in a neon city",
            face_prompts=[FacePrompt(label="Person 1", role_prompt="wearing a black coat", position="left")],
        )

        self.assertIn("cinematic portrait in a neon city.", prompt)
        self.assertIn("Main subject: wearing a black coat.", prompt)

    def test_compose_generation_prompt_for_two_faces(self) -> None:
        prompt = compose_generation_prompt(
            scene_prompt="rainy cyberpunk street",
            face_prompts=[
                FacePrompt(label="Person 1", role_prompt="holding an umbrella", position="left"),
                FacePrompt(label="Person 2", role_prompt="laughing while holding a coffee", position="right"),
            ],
            interaction_prompt="Person 1 is protecting Person 2 from the rain",
        )

        self.assertIn("Two distinct people in the same scene.", prompt)
        self.assertIn("Person 1 on the left side: holding an umbrella.", prompt)
        self.assertIn("Person 2 on the right side: laughing while holding a coffee.", prompt)
        self.assertIn("Interaction between Person 1 and Person 2: Person 1 is protecting Person 2 from the rain.", prompt)

    def test_compose_negative_prompt_adds_dual_face_safety_terms(self) -> None:
        negative = compose_negative_prompt("low quality", num_faces=2)

        self.assertIn("low quality", negative)
        self.assertIn("merged faces", negative)
        self.assertIn("duplicate person", negative)

    def test_compose_generation_prompt_compacts_structured_logo_prompt(self) -> None:
        prompt = compose_generation_prompt(
            scene_prompt="""Create a premium, high-speed logo for an F1-inspired racing company named [Company Name]. The logo should feel modern, aggressive, aerodynamic, and elite. Use sharp racing lines, motion streaks, a sleek race car silhouette, and a bold motorsport-style wordmark. The design should communicate speed, precision, power, and championship energy.

Style: futuristic motorsport, luxury racing brand, clean vector logo, minimal but powerful, suitable for car livery, merchandise, website, helmet branding, and social media.
Colors: black, red, silver, white, and carbon-fiber accents.
Typography: bold, italicized, angular, racing-inspired font.
Icon idea: abstract F1 car nose, speed trail, racing flag detail, or aerodynamic wing shape.
Avoid copying the official Formula 1 logo. Make it original, premium, and brand-ready.
Output: clean vector-style logo, transparent background, high contrast, professional branding, no mockup, no extra text.""",
            face_prompts=[],
        )

        self.assertLessEqual(len(prompt), 701)
        self.assertIn("premium, high-speed logo", prompt)
        self.assertIn("futuristic motorsport", prompt)
        self.assertIn("black, red, silver, white", prompt)
        self.assertIn("clean vector-style logo", prompt)
        self.assertNotIn("Avoid copying", prompt)
        self.assertNotIn("no mockup", prompt)

    def test_logo_prompt_derives_negative_terms_from_avoid_and_output(self) -> None:
        terms = derive_negative_prompt_terms(
            "Avoid copying the official Formula 1 logo.\n"
            "Output: clean vector-style logo, transparent background, no mockup, no extra text."
        )

        self.assertIn("official Formula 1 logo", terms)
        self.assertIn("copied F1 logo", terms)
        self.assertIn("mockup", terms)
        self.assertIn("extra text", terms)

    def test_prompt_warning_reports_placeholders(self) -> None:
        warning = compose_prompt_warning("Create a logo for [Company Name].")

        self.assertIsNotNone(warning)
        self.assertIn("[Company Name]", warning or "")


if __name__ == "__main__":
    unittest.main()
