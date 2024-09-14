# Data Mining, KSD (2024)

This course gives an introduction to the field of data mining. The course is relatively practically oriented, focusing on applicable algorithms. Practical exercises will involve both use of a freely available data mining package and individual implementation of algorithms.

## Setup the Development Environment

This repository contains the configuration files to set up the development environment. It uses `conda` to manage a Python development environment and all required packages and libraries this project uses. We use [Data Version Control (DVC)](http://dvc.org) to store all binary files (videos, images, timesheets, among others) used by examples and exercises of this course. These files are available in a private bucket on [Amazon S3](https://aws.amazon.com/s3), and the students can access them as read-only resources via an HTTPS connection. The following presents the step-by-step tutorial on setting up the `data-mining-course` development environment.

### Requirements

- An active [Anaconda](https://anaconda.org), [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install), or [Miniforge](https://github.com/conda-forge/miniforge) (highly recommended) installation with the `bin` folder added to the *Environmental Variable* `$PATH` (Linux and macOS) or `%PATH%` (Microsoft Windows).
- (Recommended) [Visual Studio Code](https://code.visualstudio.com) installed.

### Installation

First of all, use the Terminal (**Prompt**) to execute the following commands. Clone this repository and enter the project root folder:

```[bash]
git clone https://github.com/fabricionarcizo/damin2024
cd $PROJECT\damin2024
```

Then, create a new environment called `data-mining-course`:

```[bash]
conda env create -f environment.yml
```

Activate the created environment:

```[bash]
conda activate data-mining-course
```

Finally, download the binary resource available in our private Amazon S3 bucket:

```[bash]
dvc pull
```

Before each lecture, you must download the newest binary files by executing the command `dvc pull` again.

**P.S.**: Once in a while, update your development environment to get the latest pip package versions:

```[bash]
conda env update -f environment.yml --prune
```

## Description

The course will cover the following main topics:

- The data mining process
- Cluster analysis
- Data pre-processing
- Pattern and association mining
- Classification and prediction

Application examples will be given from domains including demographics, image processing and healthcare.

## Formal Prerequisites

Students must have experience with and be comfortable with programming and be capable of independently implementing algorithms from descriptions. This corresponds to passing at least an introductory programming course, preferably an intermediate-level one. The course will contain compulsory programming in [Python](http://python.org).

Students must be familiar with basic mathematical notation and concepts such as variables, sets, functions, averages, and variance. For example, a discrete mathematics course can help students acquire these competencies.

### Information about study structure

This is a specialization course for the [MSc Software Design](https://en.itu.dk/Programmes/MSc-Programmes/Software-Design) study program and an elective for other MSc study programs. Moreover, the student must always meet the admission requirements of the [IT University of Copenhagen](https://en.itu.dk).

## Intended Learning Outcomes

After the course, the student should be able to:

- Analyze data mining problems and reason about the most appropriate methods to apply to a given dataset and knowledge extraction need.
- Implement basic pre-processing, association mining, classification and clustering algorithms.
- Apply and reflect on advanced pre-processing, association mining, classification and clustering algorithms.
- Work efficiently in groups and evaluate the algorithms on real-world problems.

## Learning Activities

The course consists of lectures ending with a project for the last part of the course. Most weeks, you will have a lecture and a lab exercise involving independent programming. Students must be able to program. The default language is Python, and there is an introduction to this in Week \#01 and during the labs.

There is one mandatory assignment around the course midway, where you will apply the techniques learned.

For the final project, you will specify and work on a relevant Data Mining project of your choice. In this project, you will apply the techniques and algorithms studied during the course to relevant real-world problems. This will be done in groups of 2-4 people.

In addition to the hours planned for lectures, tutorials, and exercises, supervision sessions for the group projects are planned. These sessions complement the theory covered during the lectures and are necessary for meeting the course's learning objectives. Lectures provide theoretical foundations and walk-through examples of relevant data mining algorithms, while exercises focus on students discussing and implementing the algorithms.

The following table presents the lecture plan:
| Week | Date | Lecture |
|--|--|--|
| 01 | 30/08/2024 | [Getting Started](lecture01) |
| 02 | 06/09/2024 | [Introduction to Python Programming Language](lecture02) |
| 03 | 13/09/2024 | [Introduction to Linear Algebra](lecture03) |
| 04 | 20/09/2024 | [Data Preprocessing](lecture04) |
| 05 | 27/09/2024 | Data Exploration and Visualization |
| 06 | 04/10/2024 | Classification: Basic Concepts |
| 07 | 11/10/2024 | Classification: Advanced Techniques |
| 08 | 18/10/2024 | Regression Analysis |
| 09 | 01/11/2024 | Clustering: Basic Concepts |
| 10 | 08/11/2024 | Clustering: Advanced Techniques |
| 11 | 15/11/2024 | Dimensionality Reduction |
| 12 | 22/11/2024 | Anomaly Detection |
| 13 | 29/11/2024 | Association Rule Mining |
| 14 | 06/12/2024 | Final Exam Project and Course Evaluation |

## Mandatory Activities

One mandatory assignment is to use self-implemented data mining techniques on a simple data set and write a report about it.

The students will receive an **"Approved"**/**"Not Approved"** grade on the assignment, with follow-up formative feedback.

The pedagogical function of the mandatory project is to provide the students with an activity where they gain experiential knowledge supporting the ILOs reached in the course, including data preparation and machine learning classification.

If the students won't hand in or fail the mandatory activity, then they will have to pass a repeat mandatory examination provided within a month of the grade.

The student will receive the grade **NA** (**not approved**) at the ordinary exam, if the mandatory activities are not approved and the student will use an exam attempt.

## Course Literature

**The 100-Page Machine Learning Book**. By Andriy Burkov. Published Jan 13, 2019 by Lightning Source Inc. ISBN-10 1777005477 and ISBN-13 978-1777005474.

**Data Mining: Concepts and Techniques (The Morgan Kaufmann Series in Data Management Systems), 4th Edition**. By Jiawei Han, Micheline Kamber, and Jian Pei. Published Oct 17, 2022 by Morgan Kaufmann. ISBN-10 9780128117606 and ISBN-13 978-0128117606.

## Student Activity Budget

Estimated distribution of learning activities for the typical student:

- Preparation for lectures and exercises: 20%
- Lectures: 15%
- Exercises: 10%
- Assignments: 10%
- Project work, supervision included: 35%
- Exam with preparation: 10%

## Ordinary Exam

### Exam type

D: Submission of written work with following oral, External (7-point scale)

### Exam variation

D1G: Submission for groups with following oral exam based on the submission. Shared responsibility for the report.

### Exam submission description

The final assessment will be a jointly written report. There will be a group presentation of this report followed by individual questions about the report and the work behind it, resulting in individual grades.

### Group submission

#### Group

2-4

### Exam duration per student for the oral exam

15 minutes

### Group exam form

Mixed exam 2 : Joint student presentation followed by an individual dialogue. The group makes their presentations together and afterwards the students participate in the dialogue individually while the rest of the group is outside the room.

## Reexam

### Reexam type

B: Oral exam

### Reexam variation

B22: Oral exam with no time for preparation.

### Reexam duration per student for the oral exam

20 minutes

## Time and Date

Ordinary Exam - submission Fri, 20 Dec 2024, 08:00 - 14:00

Ordinary Exam Mon, 13 Jan 2025, 09:00 - 21:00

Ordinary Exam Tue, 14 Jan 2025, 09:00 - 21:00
