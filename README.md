# NumPy Practice Workbook

> **Learn NumPy by solving problems — not by memorizing syntax.**

A hands-on NumPy practice workbook built around a synthetic **Smart Factory Sensor Network**, with **67 progressively challenging tasks**, hidden solutions, multi-level hints, realistic data-quality problems, debugging exercises, and an integrated final challenge.

If you're learning **Python, Data Science, Machine Learning, AI Engineering, or Data Engineering**, this repository is designed to help you build the NumPy skills you'll actually use.

---

## 🚀 Why this repository?

Most NumPy tutorials teach you:

```python
np.array()
np.mean()
np.reshape()
```

…and then move on.

The problem is that knowing a function exists isn't the same as knowing **when to use it**.

This workbook is designed around a different approach:

**Learn → Attempt → Get a Hint → Solve → Verify → Test Yourself**

You won't just practice NumPy syntax.

You'll practice deciding:

* Which NumPy operation solves the problem?
* Which axis should be used?
* Will broadcasting work?
* Is this a view or a copy?
* Should this operation be vectorized?
* What happens when the data contains `NaN`?
* Which approach is more memory-efficient?
* Which NumPy function is appropriate for the problem?

---

## ✨ What makes this different?

### 🧠 67 Hands-On Tasks

The workbook contains **67 tasks** covering NumPy from fundamentals through practical, interview-style reasoning.

### 🎓 Two-Phase Learning System

#### Phase 1 — Learning

Tasks **1–38** introduce NumPy concepts progressively.

You'll practice:

* Array creation
* Array attributes
* Indexing
* Slicing
* Boolean masking
* Fancy indexing
* `np.where`
* Reshaping
* Views vs copies
* Elementwise operations
* Broadcasting
* Aggregations
* Axis-based operations
* Missing-data handling
* Sorting
* Ranking
* Concatenation
* Stacking
* Splitting
* Linear algebra
* 3D arrays
* Vectorization
* Saving/loading arrays

---

#### Phase 2 — Testing

Tasks **39–66** test whether you can actually use what you learned.

The important difference:

> **The problem generally doesn't tell you which NumPy function to use.**

You have to figure that out yourself.

The difficulty progresses from straightforward exercises to:

* Debugging
* Broadcasting failures
* Wrong-axis bugs
* Silent mutation
* Performance optimization
* Memory management
* Engineering trade-offs
* Interview-style reasoning

---

### 🏭 A Realistic Practice Dataset

Instead of random toy arrays, the workbook uses a synthetic **Smart Factory Sensor Network**.

The dataset contains:

| Data          |       Shape | Description                  |
| ------------- | ----------: | ---------------------------- |
| `temperature` | `(12, 120)` | Hourly temperature readings  |
| `vibration`   | `(12, 120)` | Hourly vibration readings    |
| `pressure`    | `(12, 120)` | Hourly pressure readings     |
| `production`  |   `(12, 5)` | Daily production per machine |
| `downtime`    |     `(12,)` | Total machine downtime       |
| `machine_ids` | `12` labels | `M01` → `M12`                |

The sensor data intentionally contains realistic imperfections:

* Missing temperature readings (`NaN`)
* Temperature fault spikes
* Vibration anomalies
* Zero-pressure wiring faults
* Machine-to-machine variation

This gives you data-quality problems to solve rather than perfectly clean textbook arrays.

---

## 💡 Interactive Hints & Solutions

The solutions are intentionally kept outside the notebook.

After loading the workbook, you can ask for hints:

```python
hint(12)
```

Need a stronger hint?

```python
hint(20, level=2)
```

Ready to see the solution?

```python
answer(12)
```

The solution system doesn't simply display the answer.

It **executes the solution against a fresh copy of the dataset and shows the actual output**, allowing you to compare your result with the expected result.

This creates a much better learning loop:

```text
Attempt
   ↓
Think
   ↓
Hint
   ↓
Try again
   ↓
Solution
   ↓
Compare output
   ↓
Understand the reasoning
```

---

## 🏆 Final Challenge

After completing the learning and testing phases, you'll face:

### Factory Health Dashboard — Integrated Challenge

You'll build a complete NumPy-based pipeline that:

1. Detects missing readings and sensor anomalies
2. Cleans the sensor data
3. Creates per-machine summaries
4. Builds a composite health score
5. Ranks machines
6. Identifies the machine most in need of maintenance
7. Justifies your result using the data

This challenge combines concepts from throughout the workbook.

---

## 📚 Skills You'll Practice

### Array Fundamentals

* `np.array`
* `np.zeros`
* `np.ones`
* `np.full`
* `np.arange`
* `np.linspace`
* `np.eye`
* Random arrays
* `shape`
* `ndim`
* `size`
* `dtype`
* `astype`
* `nbytes`

### Indexing & Selection

* 1D indexing
* 2D indexing
* Slicing
* Boolean masking
* Fancy indexing
* Coordinate-based selection
* `np.where`

### Reshaping & Memory

* `reshape`
* `flatten`
* `ravel`
* Views
* Copies
* `keepdims`

### Vectorized Computation

* Elementwise arithmetic
* Broadcasting
* Conditional operations
* Vectorization
* Loop vs vectorized performance

### Aggregation & Analysis

* `mean`
* `min`
* `max`
* `std`
* `sum`
* `any`
* `all`
* `cumsum`
* `argmax`
* `argmin`
* `argsort`
* NaN-aware operations

### Array Manipulation

* `concatenate`
* `vstack`
* `hstack`
* `split`
* `tile`
* `repeat`
* `stack`

### Linear Algebra

* Matrix multiplication
* Dot products
* Weighted calculations

### Data Quality

* Missing values
* Sensor anomalies
* Fault correction
* Validation
* Defensive copying
* Shape mismatches

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/vikasthamke/numpy-practice-workbook.git
cd numpy-practice-workbook
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
numpy_practice.ipynb
```

---

## ⚡ Quick Start

Run the setup cells in the notebook.

Then:

```python
from practice_system import hint, answer
```

For example:

```python
hint(12)
```

or:

```python
hint(20, level=2)
```

After attempting the problem:

```python
answer(12)
```

### Important

**Try the problem yourself before using `answer()`.**

The goal isn't to finish all 67 tasks as quickly as possible.

The goal is to develop the ability to look at a problem and think:

> "Which NumPy operation should I use here, and why?"

---

## 🎯 Who is this for?

This repository is especially useful for:

* Python beginners learning NumPy
* Data Science students
* Aspiring Data Scientists
* Aspiring AI Engineers
* Aspiring Data Engineers
* Machine Learning Engineers
* Students preparing for technical interviews
* Anyone who knows basic NumPy but lacks practice

You don't need advanced mathematics to start.

You should be comfortable with basic Python syntax, variables, lists, loops, and functions.

---

## 🧭 Recommended Learning Path

Don't rush through all 67 tasks.

### Step 1 — Learn

Complete Tasks **1–38**.

Use hints when necessary.

### Step 2 — Practice from memory

Redo the problems where you needed `answer()`.

### Step 3 — Test yourself

Complete Tasks **39–66** without looking up which function to use.

### Step 4 — Final Challenge

Complete Task **67** without looking at the solution first.

### Step 5 — Revisit

A few days later, redo the tasks that gave you the most trouble.

---

## 📈 What you'll gain

By the end, you should be comfortable with more than NumPy syntax.

You should be able to reason about:

```text
Shape
  ↓
Axes
  ↓
Indexing
  ↓
Broadcasting
  ↓
Vectorization
  ↓
Aggregation
  ↓
Memory
  ↓
Data quality
  ↓
Practical analysis
```

That mental model is far more valuable than memorizing individual functions.

---

## 🤝 Contributing

Found an issue?

Have a better exercise?

Think a task could be improved?

Contributions are welcome.

Some ideas:

* Additional practice tasks
* Better hints
* New datasets
* Performance-focused exercises
* More debugging challenges
* Documentation improvements
* Typo/error fixes
* New final challenges

Please open an issue before making a large change so we can discuss it first.

---

## ⭐ If this helped you

If this workbook helped you improve your NumPy skills:

**⭐ Star the repository**

It helps other learners discover the project.

If you find a mistake or have an idea for improving the exercises, open an issue or submit a pull request.

---

## 📌 Project Philosophy

This project follows one simple principle:

> **Don't just learn what a function does. Learn when and why you should use it.**

Happy practicing. 🐍

**Learn. Attempt. Debug. Understand. Repeat.**
