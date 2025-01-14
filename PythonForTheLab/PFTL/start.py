from PyQt5.QtWidgets import QApplication
from PFTL.model.experiment import Experiment
from PFTL.view.main_window import ScanWindow

def start():
    exp = Experiment()
    exp.load_config('Examples/config.yml')
    exp.initialise()

    app = QApplication([])
    win = ScanWindow(exp)  # ScanWindow has now access to exp. Experiment has config
    win.show()
    app.exec()

    exp.finalise()
    
if __name__ == '__main__':
    start()