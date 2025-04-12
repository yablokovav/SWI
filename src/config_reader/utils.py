from src import *

def create_directories(preprocessing, spectral, inversion, swi_dir) -> dict[str, list]:
    # Определение корневой директории SWI
    data_dir = preprocessing.data_dir
    if data_dir.is_dir():
        folder_name = data_dir.name
    else:
        folder_name = data_dir.parent.name

    # определение типа данных
    if (type_data := preprocessing.type_data.value) == "2d":
        save_layout = [f"offsets_{preprocessing.offset_min}_{preprocessing.offset_max}"]
    elif preprocessing.parameters_3d.sort_3d_order == 'csp':
        save_layout = [f"num_sectors_{preprocessing.parameters_3d.num_sectors}"]
    else:
        save_layout = ['cdp']

    base_dirs = [
        (f"{type_data}/preprocessing", save_layout),
        (
            f"{type_data}/spectral_analysis",
            [f"{sl}/{spectral.spectral_method.value}/{subdir}" for
             sl in save_layout for subdir in ["dc", "image", "segy"]],
        ),
        (
            f"{type_data}/inversion",
            [f"{sl}/{spectral.spectral_method.value}/{inversion.inversion_method.value}/{subdir}" for
             sl in save_layout for subdir in ["models1d_bin", "image"]],
        ),
        (
            f"{type_data}/postprocessing",
            [f"{sl}/{spectral.spectral_method.value}/{inversion.inversion_method.value}/{subdir}" for
             sl in save_layout for subdir in ["models2d_bin", "image", "segy", "fdm"]],
        ),
    ]

    # Создаем директории
    module_dirs = {"preprocessing": [], "spectral_analysis": [], "inversion": [], "postprocessing": []}
    for base, subdirs in base_dirs:
        for subdir in subdirs:
            directory = swi_dir / base / folder_name / subdir
            directory.mkdir(parents=True, exist_ok=True)
            module_dirs[base[3:]].append(directory)

    return module_dirs