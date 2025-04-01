# Behavior Tracking with Vision Language Model (VLM)

### Overview
This application is a real-time behavior tracking system using Vision Language Model (VLM) from Google's Gemini. It can analyze camera feed in real-time and detect specific behaviors based on user-defined prompts.

### Features
- Real-time camera feed analysis
- Customizable behavior detection prompts
- Visual status indicators (green/yellow/red)
- FPS and response time monitoring
- Settings persistence
- API key validation
- Horizontal camera flip option

### Demo
[Watch Demo Video](demo.mp4)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/tranvuongquocdat/Behavior-tracking-with-VLM.git
cd Behavior-tracking-with-VLM
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

1. Run the application:
```bash
python pyqt_vlm_ai.py
```

2. Configure the application:
   - Enter your Google Gemini API key in the settings panel
   - Set the update interval (in seconds)
   - Define your behavior detection prompt
   - Toggle camera flip if needed

3. Click "Save Settings" to apply changes

### Status Indicators
- Green: No behavior detected
- Yellow: Behavior detected with uncertainty
- Red: Behavior detected with high confidence