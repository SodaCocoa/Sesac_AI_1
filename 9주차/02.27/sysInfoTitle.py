import sys

class Line:
    def __init__(self, char='-', count=100):
        self.char = char
        self.count = count
        
    def LinePrn(self, char=None, count=None):
        if char is None:
            char = self.char
        if count is None:
            count = self.count
        return '\n▶' + char * count
        
class PythonTitlePrinter(Line):
    def sysInfo(self):         
        info = []  # Assuming you intended to use this for something
        info.append(self.LinePrn())
        info.append("▷ Python Version: " + sys.version)
        info.append("▷ Python Implementation: " + sys.implementation.name)
        info.append("▷ Python Compiler: " + sys.version.split()[3])
        info.append("▷ Python Build Number: " + sys.version.split()[2])
        info.append(self.LinePrn('#', 100))
        return "\n".join(info)

# 예제 사용 (Example usage)
python_title_printer = PythonTitlePrinter()
print(python_title_printer.sysInfo())
