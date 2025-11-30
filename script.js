document.getElementById('runCode').addEventListener('click', function() {
    let code = document.getElementById('code').value;
    try {
        let result = eval(code);
        document.getElementById('output').innerText = "Výstup: " + result;
    } catch (e) {
        document.getElementById('output').innerText = "Chyba: " + e.message;
    }
});
