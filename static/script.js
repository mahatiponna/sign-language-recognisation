
const alphabets = "ABCDEFGHIKLMNOPQRSTUVWXY";

const alphabetGrid = document.getElementById("alphabetGrid");

alphabets.split("").forEach(letter => {
    const item = document.createElement("div");
    item.textContent = letter;
    alphabetGrid.appendChild(item);
});
