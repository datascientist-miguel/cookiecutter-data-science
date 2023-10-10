# Cookiecutter Project

## Requirements

- [Anaconda](https://www.anaconda.com/download/) >= 4.x
- [git](https://git-scm.com/)
- [Cookiecutter Python package](http://cookiecutter.readthedocs.org/en/latest/installation.html)

``` bash
conda install -c conda-forge cookiecutter
```

## Create a new project

In the directory where you want to create the project, run:

```bash
cookiecutter .
```

or

```bash
cookiecutter https://github.com/datascientist-miguel/cookiecutter-data-science
```

You will be asked about your project name, author, etc. This information will be used to create the project.


## Estructure of the project

```bash
{{ cookiecutter.project_slug }}
├── LICENCE
├── README.md
├── environment.yml
├── .gitignore
├── data
│   ├── raw
│   └── processed
├── docs
├── models
├── notebooks
│   └── 0.0-{{ cookiecutter.project_slug }}-introduction.ipynb
|   └── 0.1-{{ cookiecutter.project_slug }}-eda.ipynb
|   └── 0.2-{{ cookiecutter.project_slug }}-visualization.ipynb
|   └── 0.3-{{ cookiecutter.project_slug }}-modeling.ipynb
├── report
├── scripts
│   ├── analysis
│   ├── manipulation
│   ├── modeling
│   ├── visualization
```


## Credits

This project was made it in base of:

Platzi Configuración Profesional de Entorno de Trabajo para Ciencia de Datos course by Jesús Vélez Santiago
Cookiecutter Data Science repository by Driven Data