# SWI

## Описание
Программный модуль предназначен для обработки сейсмических данных методом SWI (Surface Waves Inversion) и предоставляет комплексный инструмент реализующий все этапы (препроцессинг, спектральный анализ, инверсия и постпроцессинг) обработки методом SWI  для построения 2D и 3D скоростных моделей S-волны.

Документация (руководство пользователя, техническое описание и перечень управляемых параметров) доступна по ссылке:
https://docs.google.com/document/d/1zbgvVUEJEtaVe5a9OVKEFZOM0MUto0jHu3e32z1EUN0/edit?tab=t.0

## Установка
1. Скачать архив:
   - в Windows - распаковать с помощью файлового менеджера(необходим 7-Zip или WinRar) или через терминал:
   ```PowerShell
   "C:\Program Files\7-Zip\7z.exe" x SWI-main.7z
   ```
   заменить "C:\Program Files\7-Zip\7z.exe" на свой путь до исполняемого файла 7-Zip при необходимости
   - в Linux - для успешной установки необходимы следующие компоненты 7-Zip, build-essential, tk-dev, libffi-dev, tcl-dev, libssl-dev(для установки дополнительных библиотек при необходимости):
   распаковать архив:
   ```bash
   unzip SWI-main.zip
   ```

2. Перейдите в директорию проекта:
   ```bash
   cd SWI-main
   ```

3. Установите питон:
   - в Windows - зуапустите файл python-3.10.11-amd64.exe:
   ```PowerShell
   python-3.10.11-amd64.exe /quiet InstallAllUsers=1 AssociateFiles=1 Include_doc=1 Include_pip=1 Include_test=0 AddPythonToPath=1       InstallDir="C:\Users\<User>\AppData\Local\Programs\Python\Python310"
   ```
   перезпустите терминал
   - в Linux - распакуйте архив Python-3.10.11.tgz:
   ```bash
   tar -xzf Python-3.10.11.tgz
   ```
   перейтиде в распакованный каталог
   ```bash
   cd Python-3.10.11
   ```
   сконфигурируйте сбрку:
   ```bash
   ./configure --prefix=/opt/python3.10 --enable-optimizations
   ```
   соберите python
   ```bash
   make -j$(nproc)
   ```
   установите python
   ```bash
   sudo mkdir -p /opt/python3.10
   sudo chown $USER:$USER /opt/python3.10
   sudo make install
   ```
   добавьте в PATH:
   - откройте файл ~/.bashrc (или ~/.bash_profile, ~/.zshrc) и добавьте следующие строки в конец файла:
   ```bash
   export PATH="/opt/python3.10/bin:$PATH"
   export LD_LIBRARY_PATH="/opt/python3.10/lib:$LD_LIBRARY_PATH"
   ```
   - сохраните файл и обновите оболочку:
   ```bash
   source ~/.bashrc  # Или source ~/.bash_profile, source ~/.zshrc
   ```
4. Проверка установки: 
   - в Windows:
   ```PowerShell
   python --version
   ```
   - в Linux:
   ```bash
   python3.10 --version
   ```
   должна отобразится версия 3.10.11

5. В каталоге с проектом создайте виртуальную среду Python:
   - в Windows:
   ```PowerShell
   python -m venv <venv_name>
   ```
   - в Linux:
   ```
   python3.10 -m venv <venv_name>
   ```
7. Активируте виртуальную среду:
   - в Windows:
   ```PowerShell
   <venv_name>\Scripts\activate
   ```
   - в Linux:
   ```bash
   source <venv_name>/bin/activate
   ```
8. Установите все необходимые пакеты из локального каталога:
   - в Windows:
   ```PowerShell
   pip install --no-index --find-links=./packages_for_windows -r .\requirements_for_windows.txt
   ```
   - в Linux:
   ```bash
   pip install --no-index --find-links=./packages_for_linux -r requirements_for_linux.txt
   ```

## Запуск обработки
Запуск обработки по этапам осуществляется через команду python 

1. Запуск препроцессинга и спектрального анализа:
   ```bash
   python _1_preprocessing_spectral_run.py
   ```
2. Запуск инверсии:
     ```bash
     poetry _2_inversion_run.py
     ```
2. Запуск постпроцессинга:
     ```bash
     poetry _3_postprocessing_run.py
     ```
