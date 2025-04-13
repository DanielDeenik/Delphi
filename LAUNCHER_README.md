# Delphi Trading Intelligence System Launcher

This document explains how to use the launcher scripts to automatically start the Delphi Trading Intelligence System.

## Quick Start

### Cross-Platform Launcher (Recommended)

```bash
# Python launcher (works on all platforms)
python launch_delphi.py
```

### Simple Wrapper Scripts

```bash
# Windows
launch.bat

# Unix/Linux/macOS
./launch.sh
```

### Platform-Specific Scripts

```bash
# Windows
launch_delphi.bat

# Unix/Linux/macOS
./launch_delphi.sh
```

## What the Launcher Does

All launcher scripts perform the following steps:

1. **Sets up the environment**:
   - Creates necessary directories (logs, status, templates, static, config)
   - Creates a default configuration file if one doesn't exist

2. **Starts the application**:
   - Launches the Flask application on port 3000
   - Redirects output to a log file

3. **Opens the browser**:
   - Waits for the application to start
   - Opens the browser to the multi-tab Colab view (http://localhost:3000/colab/all)

4. **Displays information**:
   - Shows URLs for accessing the application
   - Provides instructions for stopping the application

## Accessing the Application

Once the launcher has started the application, you can access it at the following URLs:

- **Dashboard**: http://localhost:3000
- **Notebooks**: http://localhost:3000/colab
- **All Notebooks**: http://localhost:3000/colab/all

## Stopping the Application

### Windows

Press Ctrl+C in the command window, then close the window.

### Unix/Linux/macOS

Press Ctrl+C in the terminal window. The launcher script will automatically stop the application.

## Troubleshooting

### Application Fails to Start

If the application fails to start, check the log file in the `logs` directory. The log file is named `app_YYYYMMDD.log` (where YYYYMMDD is the current date).

### Browser Doesn't Open

If the browser doesn't open automatically, you can manually open it and navigate to http://localhost:3000.

### Port Already in Use

If port 3000 is already in use, you can edit the launcher script to use a different port. Look for the line that contains `--port 3000` and change it to a different port number.

## Advanced Usage

### Cross-Platform Launcher

The `launch_delphi.py` script is a cross-platform launcher that works on Windows, macOS, and Linux. It uses Python's built-in libraries to detect the operating system and launch the application accordingly.

Advantages of the cross-platform launcher:
- Works on all platforms
- Better error handling
- Cleaner code
- Easier to maintain

### Customizing the Configuration

The launcher creates a default configuration file if one doesn't exist. You can customize this configuration by editing the `config/config.json` file.

### Running with Different Options

If you need to run the application with different options, you can edit the launcher script or run the application manually:

```bash
# Run with debug mode
python -m trading_ai.cli.dashboard_cli --port 3000 --debug

# Run on a different port
python -m trading_ai.cli.dashboard_cli --port 8080
```

### Updating Data Before Launch

If you want to update the data before launching the application, you can run the data import scripts first:

```bash
# Windows
scripts\ticker_imports\import_master.bat

# Unix/Linux/macOS
./scripts/ticker_imports/import_master.sh
```

Then run the launcher script to start the application.
