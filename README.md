# <center>Detection API</center>

---

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
        <li><a href="#technologies">Technologies</a></li>
    <li>
      <a href="#setting-up-the-project">Setting Up the Project</a>
      <ul>
      </ul>
    </li>
    <li><a href="#contribuitons">Contributions</a></li>

  </ol>
</details>

### About the project

This is a bachelor project at NTNU in course [IDATA2900][IDATA2900-url]. The Project is develop based on a company´s description and desire.

We have had regular meetings with client and school supervisor to have a steady progress in the throghout the development.

This repository is the picture recognition component.
Main functionality is to recognise the objects from the imported pictures. This happen through Open Computer Vision.

---

### Technologies

Open CV
Yolo v8
Fast API - Python
Uvicorn - Python

#### Requirements

Python 3.10+

---

### Setting Up the Project

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd detection-api
```

#### 2. Generate the virtual enviroment

```bash
python3 -m venv .venv
```

#### 3. Install the requirements

```bash
pip install -r requirements.txt
```

#### 4. Activate the enviroment

```bash

source .venv/bin/activate
```

#### 5. Start the application

To build and run the application:

```bash
uvicorn main:app --host <IP> --port <port> --reload
```

---

### Contributions

[Contributors][contributors-url]

[Viljar Hoem-Olsen][Viljar-url]
[Sander Grimstad][Sander-url]
[Thomas Aakre][Thomas-url]

---

[IDATA2900-url]: https://www.ntnu.no/studier/emner/IDATA2900/2024#tab=omEmnet
[Endpoint Badge]: https://img.shields.io/endpoint
[contributors-url]: https://github.com/Bachelor-Group-13/inneparkert-backend/graphs/contributors
[Backend-url]: https://github.com/Bachelor-Group-13/inneparkert-backend.git
[Repository-url]: https://github.com/Bachelor-Group-13
[Viljar-url]: https://github.com/viljarh
[Sander-url]: https://github.com/sagrimstad
[Thomas-url]: (https://github.com/thomasaakre)
