import os
import subprocess


def puml_to_png(puml_filename, extra_args=None):
    """
    Converts a PlantUML file to a PNG image.

    Prefers a local plantuml.jar (same directory as this script) because the
    bundled AWSCommon.puml requires PlantUML >= 2024. Falls back to the system
    `plantuml` command if the jar is not present.

    Download the jar from: https://github.com/plantuml/plantuml/releases

    Args:
        puml_filename (str): The path to the input PlantUML (.puml) file.
        extra_args (list): Optional extra arguments (e.g. ["-DRELATIVE_INCLUDE=relative"]).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jar_path = os.path.join(script_dir, "plantuml.jar")
    out_dir = os.path.dirname(os.path.abspath(puml_filename))

    if os.path.exists(jar_path):
        cmd = ["java", "-jar", jar_path, "-tpng", "-o", out_dir] + (extra_args or []) + [puml_filename]
    else:
        cmd = ["plantuml", "-tpng", "-o", out_dir] + (extra_args or []) + [puml_filename]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"OK: {os.path.basename(puml_filename)}")
        else:
            print(f"ERROR: {os.path.basename(puml_filename)}\n{result.stderr}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # AWS infrastructure diagram (uses local assets/aws-plantuml/)
    puml_to_png(os.path.join(script_dir, "aws-infrastructure.puml"))

    # C4 diagrams (use -DRELATIVE_INCLUDE so C4-PlantUML picks up assets/c4-plantuml/)
    c4_dir = os.path.join(script_dir, "c4")
    for name in ("context.puml", "container.puml", "component.puml"):
        puml_to_png(os.path.join(c4_dir, name), extra_args=["-DRELATIVE_INCLUDE=relative"])
