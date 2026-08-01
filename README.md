This repository contains a port of karpathy/llama2.c (run.c + runq.c) for EDK2 (EFI), with a focus on chatting with the model. The EFI application is written for x64 in debug mode.
Chatting capabilities:
  1) Execute commands to open a file or display `secret.txt` (password-protected): use the `<CMD>/<CMD>` marker
  2) Determine the PC’s configuration and send it to the model
  3) Exit the application—on demand or when the model gets very angry
  4) Adjust model parameters such as `/temp`, `/top_p`, etc.
  5) Cyrillic support in the EFI shell. It runs as a bootloader application (the application renamed in bootx64 to efi/boot/), but I haven’t adapted the screen resolution yet (it will appear as “mode 80” in the EFI shell)
  6) Multithreading support—all CPU threads are utilized. If there’s a bug in the firmware—the MP core doesn’t start—single-threaded mode is enabled.
  7) Support for Q_8 (INT) models. Using a custom model (Zlobych) with 200M parameters as an example—the weight size is ≈200 Mb

Tested on QEMU (MP works correctly). On a real PC: AMD Ryzen 5 Pro U version (16 threads/6 cores) (16 GB RAM)—works great, very fast. AMD E1-2500 (2/2) (4 GB RAM)—intermittent issues with MP; performance on a single core is below average, but it works. 

---

Данный репозиторий представлен портом karpathy/llama2.c (run.c + runq.c) под EDK2 (EFI) c уклоном в чатинг с моделью. Приложение efi написано под x64 в дебаг режиме.
Возможности чатинга:
  1) Выполнять команды открытие файла или вывода secret.txt (по паролю): маркёр `<CMD>/<CMD>`
  2) Определение конфигурации ПК с отправкой в модель
  3) Выход из приложения - по просьбе или когда сильно разозлить модель
  4) Регулировка параметров модели /temp, /top_p и т.д
  5) Поддержка кириллицы в среде efi shell. Как приложение-загрузчик (переименованное приложение в bootx64 в efi/boot/) запускается но пока не адаптировал разрешение экрана (там будет как mode 80 в efi shell)
  6) Поддержка многопоточного режима - используются все потоки процессора. При баге в прошивке - не запускается MP ядро - включается однопоток.
  7) Работа с Q_8 (INT) моделей. На примере кастомной модели (Злобыч) с 200М параметров - размер весов ≈200 Mb

Тестировалось на QEMU (корректно работает MP). На реальноп ПК: Amd Ryzen 5 Pro U version (16 потоков/6 ядер) (16gb RAM) - отлично работает, шустро. Amd E1-2500 (2/2)(4gb RAM) - перебои с MP, на одном ядре хуже среднего, но работает. 
