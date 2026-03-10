import os
import subprocess


def puml_to_png(puml_filename):
    """
    Converts a PlantUML file to a PNG image using a local plantuml.jar.

    Args:
        puml_filename (str): The path to the input PlantUML (.puml) file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jar_path = os.path.join(script_dir, "plantuml.jar")
    outfile = os.path.splitext(puml_filename)[0] + '.png'

    try:
        result = subprocess.run(
            ["java", "-jar", jar_path, "-tpng", "-o",
             os.path.dirname(os.path.abspath(outfile)), puml_filename],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"Successfully generated PNG: {outfile}")
        else:
            print(f"An error occurred:\n{result.stderr}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    puml_to_png(os.path.join(script_dir, "aws-infrastructure.puml"))
