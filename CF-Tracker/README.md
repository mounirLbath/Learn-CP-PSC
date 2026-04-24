# CF-Tracker

## Setup

Before running the project, place the following dataset (codeforces_submissions.csv) at the **root of the project**:

```text
https://drive.google.com/file/d/13T1sgvh869RIhvfCuyPEGsI7QJBJdz0L/view?usp=sharing
```

This file is required for the project to work, but it is not included in the GitHub repository because it is too large.

The project structure should look like this:

```text
project-root/
├── codeforces_submissions.csv
├── manage.py
├── ...
```

## Running the project

From the root of the project, run:

```bash
python manage.py runserver
```

Then open the local server URL shown in the terminal, usually:

```text
http://127.0.0.1:8000/
```

## Using the recommender

To generate ML-based recommendations:

1. Open the website.
2. Click the button at the top of the page.
3. Wait for the recommendations to be generated.

The recommendation process can take a few minutes.

**Do not click the button repeatedly while waiting**, as the computation may still be running.