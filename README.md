# NCA-GENL Prep Course

A complete, self-paced preparation course for the
**NVIDIA-Certified Associate: Generative AI LLMs (NCA-GENL)** exam, built as a
MkDocs Material documentation site and published to GitHub Pages.

The course is structured 1:1 against the [official NVIDIA exam study guide](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/):

| Domain | Weight |
| --- | --- |
| 1. Core Machine Learning and AI Knowledge | 30% |
| 2. Software Development | 24% |
| 3. Experimentation | 22% |
| 4. Data Analysis | 14% |
| 5. Trustworthy AI | 10% |

It contains study notes, per-domain quizzes, two full weighted mock exams and
seven hands-on Python labs.

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve
```

Then open <http://127.0.0.1:8000>.

## Publish

Push to `main`. The GitHub Actions workflow in `.github/workflows/docs.yml`
builds the site and deploys it to the `gh-pages` branch. Enable
**Settings → Pages → Deploy from a branch → `gh-pages` / (root)** once.

## Disclaimer

Unofficial study material. Not affiliated with or endorsed by NVIDIA. All
exam objectives are paraphrased from NVIDIA's publicly published study guide;
practice questions are original and are **not** real exam questions.
