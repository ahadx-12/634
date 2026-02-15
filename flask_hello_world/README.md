# Flask Hello World

A simple Flask application that returns "Hello, World!".

## Setup

1.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use .venv\Scripts\activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the App

Run the application using the following command:

```bash
flask --app app run
```

The application will be available at `http://127.0.0.1:5000`.

## Running Tests

Run the tests using `pytest`:

```bash
pytest
```
