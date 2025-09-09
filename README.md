# count_digits

This project is a simple digit counter task given as part of an optional test.  
It processes a folder of handwritten digit images (0–9), predicts each digit,  
and outputs the total count of each class into a CSV file.

---

## 🚀 Features
- Reads all images in a given folder (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tif`, `.tiff`)
- Preprocesses images: grayscale → padding → resize → Otsu thresholding → 28×28 vector
- Uses a small **K-Nearest Neighbors** classifier trained on MNIST (subset)
- Outputs:
  - **10-element array** `[0_count, 1_count, ..., 9_count]`
  - A `digit_counts.csv` file with counts
- Optional `--preview` flag shows sample file → prediction mappings

---

## 📦 Requirements
- Node.js 20+ (tested on Node.js 22)
- Dependencies:
  ```bash
  npm install

---

## Installed libraries:
- sharp – image preprocessing
- glob – file matching
- mnist – training dataset
- (No external ML libs needed, includes a lightweight KNN)

---

## 🔧 Usage
1. Clone the repo
- git clone https://github.com/bazanadriana/count_digits.git
- cd count_digits
2. Install dependencies
- npm install
3. Run the digit counter
- node count_digits.js /path/to/digits --preview

---

## 📂 Output
- Console shows the 10-element counts and total number of files processed.
- A file digit_counts.csv is saved in the project root:
0,1,2,3,4,5,6,7,8,9
1353,1638,826,992,1563,1210,1067,866,999,1486

---

## 📝 License
- This project is provided for test/demo purposes only.

---

## Author
Adriana Bazan
GitHub: @bazanadriana