import sys, os
sys.path.insert(0, os.path.dirname(__file__))

"""
face-scanner — holographic face point cloud visualizer
run: python main.py
"""

from src.scanner import FaceScanner

if __name__ == "__main__":
    scanner = FaceScanner()
    scanner.run()
