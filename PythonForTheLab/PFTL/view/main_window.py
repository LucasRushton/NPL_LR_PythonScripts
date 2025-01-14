from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog
from PyQt5.QtCore import QTimer
from PyQt5 import uic  # user interface module

from pathlib import Path  # used this before for file paths, very useful

class ScanWindow(QMainWindow):
    def __init__(self, experiment):
        super().__init__()  # This is how we ALWAYS need to deal with Qt, we need super().__init__() which is essential
        view_folder = Path(__file__).parent
        uic.loadUi(str(view_folder / 'scan_window.ui'), self)
        
        self.experiment = experiment  # ScanWindow now has access to experiment!
        self.plot = self.plot_widget.plot(self.experiment.voltages, self.experiment.currents)

        self.start_button.clicked.connect(self.button_pressed)
        self.stop_button.clicked.connect(self.experiment.stop_scan)
        self.save_button.clicked.connect(self.save_data)
        self.plot_button.clicked.connect(self.button_pressed_plot)
        
        self.start_line.setText(f"{self.experiment.config['Scan']['start']}")
        self.stop_line.setText(f"{self.experiment.config['Scan']['stop']}")
        self.step_line.setText(f"{self.experiment.config['Scan']['step']}")

        self.channel_in_line.setText(f"{self.experiment.config['Scan']['channel_in']}")
        self.channel_out_line.setText(f"{self.experiment.config['Scan']['channel_out']}")

        self.plot = self.plot_widget.plot(self.experiment.voltages, self.experiment.currents)
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(50)
        
        self.actionStart.triggered.connect(self.button_pressed)
        self.actionSave.triggered.connect(self.save_data)
        
    #def save_data(self):
    #    self.experiment.save_data()
        
    def button_pressed(self):
        print('Button Clicked')
        self.experiment.config['Scan'].update(
            {'start': float(self.start_line.text()),
             'stop': float(self.stop_line.text()),
             'step': float(self.step_line.text()),
             'channel_in': int(self.channel_in_line.text()),
             'channel_out': int(self.channel_out_line.text())
             }
        )
        self.experiment.start_scan()
        
    def update_plot(self):
        self.plot.setData(self.experiment.voltages, self.experiment.currents)
        
    def update_ui(self):
        if self.experiment.scan_running:
            self.start_button.setEnabled(False)
        else:
            self.start_button.setEnabled(True)
        
    def save_data(self):
        print('Save button clicked')
        directory = QFileDialog.getExistingDirectory(self, "Choose Directory", 
                                                     self.experiment.config['Save']['data_folder'])
        self.experiment.config['Save'].update({
            'data_folder': directory,
        })
        self.experiment.save_data()
        
    def button_pressed_plot(self):
        print('Plot button clicked')
        #self.experiment.scan_voltages()
        self.experiment.make_plot()
        
if __name__ == '__main__':
    app = QApplication([])  # Constantly checking user interface, checking mouse, checking size of window
    win = ScanWindow()
    win.show()  # Same as plt.show. Matplotlib is using Qt
    app.exec()