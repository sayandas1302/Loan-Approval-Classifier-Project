function updateSliderValue(sliderId, valueId) {
    var slider = document.getElementById(sliderId);
    var value = document.getElementById(valueId);
    value.textContent = slider.value;
}

//function showDiv() {
//    var div = document.getElementById("myDiv");
//    div.style.display = "block";
//}