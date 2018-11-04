import research_toolbox.tb_filesystem as tb_fs
import research_toolbox.tb_io as tb_io
import subprocess


### project manipulation
def create_project_folder(folderpath, project_name, initialize_git_repo=False):
    fn = lambda xs: tb_fs.join_paths([folderpath, project_name] + xs)

    tb_fs.create_folder(fn([]))
    # typical directories
    tb_fs.create_folder(fn([project_name]))
    tb_fs.create_folder(fn(["analyses"]))
    tb_fs.create_folder(fn(["data"]))
    tb_fs.create_folder(fn(["experiments"]))
    tb_fs.create_folder(fn(["notes"]))
    tb_fs.create_folder(fn(["temp"]))

    # code files (in order): data, preprocessing, model definition, model training,
    # model evaluation, main to generate the results with different relevant
    # parameters, setting up different experiments, analyze the results and
    # generate plots and tables.
    tb_fs.create_file(fn([project_name, "__init__.py"]))
    tb_fs.create_file(fn([project_name, "data.py"]))
    tb_fs.create_file(fn([project_name, "preprocess.py"]))
    tb_fs.create_file(fn([project_name, "model.py"]))
    tb_fs.create_file(fn([project_name, "train.py"]))
    tb_fs.create_file(fn([project_name, "evaluate.py"]))
    tb_fs.create_file(fn([project_name, "main.py"]))
    tb_fs.create_file(fn([project_name, "experiment.py"]))
    tb_fs.create_file(fn([project_name, "analyze.py"]))

    # add an empty script that can be used to download data.
    tb_fs.create_file(fn(["data", "download_data.py"]))

    # common notes to keep around.
    tb_fs.create_file(fn(["notes", "journal.txt"]))
    tb_fs.create_file(fn(["notes", "reading_list.txt"]))
    tb_fs.create_file(fn(["notes", "todos.txt"]))

    # placeholders
    tb_io.write_textfile(
        fn(["experiments", "readme.txt"]),
        ["All experiments will be placed under this folder."])

    tb_io.write_textfile(
        fn(["temp", "readme.txt"]), [
            "Here lie temporary files that are relevant or useful for the project "
            "but that are not kept under version control."
        ])

    tb_io.write_textfile(
        fn(["analyses", "readme.txt"]), [
            "Here lie files containing information extracted from the "
            "results of the experiments. Tables and plots are typical examples."
        ])

    # typical git ignore file.
    tb_io.write_textfile(
        fn([".gitignore"]),
        ["data", "experiments", "temp", "*.pyc", "*.pdf", "*.aux"])

    if initialize_git_repo:
        subprocess.call(
            "cd %s && git init && git add -f .gitignore * && "
            "git commit -a -m \"Initial commit for %s.\" && cd -" % (fn(
                []), project_name),
            shell=True)
