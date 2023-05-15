// function generateCheckboxes() {
//     const checkboxesDiv = document.getElementById('checkboxes');
//     let sentence = document.getElementById('sentence').textContent;
//
//     fetch('/generate_checkboxes')
//         .then(res => res.json())
//         .then(data => {
//             // Update the sentence text
//             sentence = data.sentence;
//             document.getElementById('sentence').textContent = sentence;
//
//             // Update the checkboxes
//             checkboxesDiv.innerHTML = '';
//             for (const checkbox of data.checkboxes) {
//                 const label = document.createElement('label');
//                 const input = document.createElement('input');
//                 input.type = 'checkbox';
//                 input.name = 'word';
//                 input.value = checkbox.word;
//                 label.appendChild(input);
//                 label.appendChild(document.createTextNode(' ' + checkbox.word));
//                 checkboxesDiv.appendChild(label);
//                 checkboxesDiv.appendChild(document.createElement('br'));
//             }
//         })
//         .catch(error => console.error(error));
// }