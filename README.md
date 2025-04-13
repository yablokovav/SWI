# SWI

## Описание
Данный проект реализует модуль обработки сейсмических данных методом SWI (Sufrace Waves Inversion).
Программный модуль SWI позволяет обрабатывать 2D и 3D данные сейсморазведки и строить 2D и 3D модели скорости S-волны.
Обработка включает этапы препроцессинга, спектрального анализа, инверсии и построцессинга.

Перечень и описание входных параметров доступно по ссылке:
https://clck.ru/3LDvC9

## Установка
1. Скачать архив:
   - в Windows - распаковать с помощью файлового менеджера(необходим 7-Zip или WinRar) или через терминал
   ```PowerShell
   "C:\Program Files\7-Zip\7z.exe" x SWI.7z
   ```
   заменить "C:\Program Files\7-Zip\7z.exe" на свой путь до исполняемого файла 7-Zip при необходимости
   - в Linux - при необходимости установить 7zip:
   ```bash
   sudo apt-get update
   sudo apt-get install unzip
   ```
   затем распаковать архив:
   ```bash
   unzip SWI.zip
   ```

2. Перейдите в директорию проекта:
   ```bash
   cd SWI
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
   make install
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
   python -- version
   ```
   - в Linux:
   ```bash
   python3.10 --version
   ```
   должна отобразится версия 3.10.11

5. В каталоге с проектом создайте виртуальную среду:
   ```bash
   python -m venv <venv_name>
   ```
6. Активируте виртуальную среду:
   - в Windows:
   ```PowerShell
   <venv_name>\Scripts\activate
   ```
   - в Linux:
   ```bash
   source <venv_name>/bin/activate
   ```
7. Установите все необходимые пакеты из окального каталога:
   ```bash
   pip install --no-index --find-links=./packages -r requirements.txt
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
