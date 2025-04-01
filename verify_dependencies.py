#!/usr/bin/env python3
"""
Script para verificar la compatibilidad de las dependencias con Python 3.9.
Este script toma requirements-lambda.txt e intenta determinar las versiones más recientes
de cada dependencia que son compatibles con Python 3.9.
"""

import json
import re
import subprocess
import sys


def check_package_versions(package_name):
    """Verifica qué versiones de un paquete están disponibles y sus requisitos de Python."""
    print(f"Verificando versiones disponibles para {package_name}...")

    try:
        # Obtener información de versiones usando pip index
        result = subprocess.run(
            ["pip", "index", "versions", package_name],
            capture_output=True,
            text=True,
            check=True,
        )

        output = result.stdout

        # Extraer versiones y requisitos de Python
        versions = []
        for line in output.splitlines():
            # Buscar líneas con información de versión
            if "Available versions:" in line:
                version_line = line.split("Available versions:")[1].strip()
                version_matches = re.findall(r"\b\d+\.\d+\.\d+\b", version_line)
                versions.extend(version_matches)

        return versions
    except subprocess.CalledProcessError as e:
        print(f"Error al obtener información para {package_name}: {e}")
        return []


def check_python_compatibility(package_name, version):
    """Verifica si una versión específica de un paquete es compatible con Python 3.9."""
    try:
        # Usar pip para verificar compatibilidad
        result = subprocess.run(
            ["pip", "install", f"{package_name}=={version}", "--dry-run"],
            capture_output=True,
            text=True,
        )

        # Si hay un error que menciona Python, probablemente no sea compatible
        if "Requires-Python" in result.stderr and "3.9" not in result.stderr:
            return False
        return True
    except Exception as e:
        print(f"Error verificando {package_name}=={version}: {e}")
        return False


def find_compatible_versions(package_name, count=5):
    """Encuentra las versiones más recientes compatibles con Python 3.9."""
    all_versions = check_package_versions(package_name)

    if not all_versions:
        return []

    # Ordenar versiones (más recientes primero)
    all_versions.sort(key=lambda v: [int(x) for x in v.split(".")], reverse=True)

    compatible_versions = []
    for version in all_versions[:10]:  # Verificar las 10 versiones más recientes
        if check_python_compatibility(package_name, version):
            compatible_versions.append(version)
            if len(compatible_versions) >= count:
                break

    return compatible_versions


def main():
    """Función principal."""
    # Leer requirements-lambda.txt
    with open("requirements-lambda.txt", "r") as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

    print("Analizando requisitos para compatibilidad con Python 3.9...")

    results = {}
    for req in requirements:
        # Separar nombre del paquete y versión
        parts = req.split("==")
        package_name = parts[0]

        compatible_versions = find_compatible_versions(package_name)

        if compatible_versions:
            results[package_name] = compatible_versions
            print(
                f"✅ {package_name}: Versiones compatibles con Python 3.9: {', '.join(compatible_versions)}"
            )
        else:
            print(
                f"❌ {package_name}: No se encontraron versiones compatibles con Python 3.9"
            )

    # Guardar resultados
    with open("compatible_versions.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResultados guardados en compatible_versions.json")
    print("\nSugerencias para requirements-lambda.txt:")
    for package, versions in results.items():
        if versions:
            print(f"{package}=={versions[0]}")


if __name__ == "__main__":
    main()
