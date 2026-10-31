"""
train_distilbert.py - Orquestador principal del pipeline de DistilBERT
Ejecuta 01_eda.py, 02_train.py y 03_error_analysis.py en secuencia.
"""
import subprocess
import sys

def run_step(script_name):
    print("\n" + "=" * 65)
    print(f" EJECUTANDO: {script_name}")
    print("=" * 65)
    ret = subprocess.run([sys.executable, script_name])
    if ret.returncode != 0:
        print(f"[ERROR] Falló la ejecución de {script_name}. Abortando.")
        sys.exit(ret.returncode)

if __name__ == "__main__":
    print("Iniciando pipeline integral de DistilBERT sobre SQuAD 2.0...")
    run_step("pipeline_distilbert/01_eda.py")
    run_step("pipeline_distilbert/02_train.py")
    run_step("pipeline_distilbert/03_error_analysis.py")
    print("\n" + "=" * 65)
    print(" ¡PIPELINE DE DISTILBERT COMPLETADO EXITOSAMENTE!")
    print(" Revisa todos los artefactos generados en logs_distilbert/")
    print("=" * 65)