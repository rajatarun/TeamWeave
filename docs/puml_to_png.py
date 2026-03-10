import os
import subprocess


def puml_to_png(puml_filename, extra_args=None):
    """
    Converts a PlantUML file to a PNG image using a local plantuml.jar.

    Args:
        puml_filename (str): The path to the input PlantUML (.puml) file.
        extra_args (list): Optional extra arguments passed to plantuml (e.g. ["-DRELATIVE_INCLUDE=relative"]).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jar_path = os.path.join(script_dir, "plantuml.jar")
    out_dir = os.path.dirname(os.path.abspath(puml_filename))

    cmd = ["java", "-jar", jar_path, "-tpng", "-o", out_dir] + (extra_args or []) + [puml_filename]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Successfully generated PNG for: {os.path.basename(puml_filename)}")
        else:
            print(f"An error occurred:\n{result.stderr}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # AWS infrastructure diagram (uses local aws-plantuml-dist/)
    puml_to_png(os.path.join(script_dir, "aws-infrastructure.puml"))

    # C4 diagrams (use -DRELATIVE_INCLUDE so C4-PlantUML picks up local c4-plantuml-dist/)
    c4_dir = os.path.join(script_dir, "c4")
    for name in ("context.puml", "container.puml", "component.puml"):
        puml_to_png(os.path.join(c4_dir, name), extra_args=["-DRELATIVE_INCLUDE=relative"])
