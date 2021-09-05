var div1 = document.getElementById("Perfil");
var div2 = document.getElementById("Temp");
var div3 = document.getElementById("Acel");
var div4 = document.getElementById("Cam");
var div5 = document.getElementById("editPerfil");
div5.style.display = 'none';
div4.style.display = 'none';
div3.style.display = 'none';
div2.style.display = 'none';
div1.style.display = 'none';

function turnOff(div){
    if (div.style.display !== 'none'){
        div.style.display = 'none';
    }
}

function turnOn(div){
    if (div.style.display == 'none'){
        div.style.display = 'block';
    }
}

function clk1(){
    turnOn(div1);
    turnOff(div2);
    turnOff(div3);
    turnOff(div4);
    turnOff(div5);
}

function clk2(){
    turnOn(div2);
    turnOff(div1);
    turnOff(div3);
    turnOff(div4);
    turnOff(div5);
}

function clk3(){
    turnOn(div3);
    turnOff(div1);
    turnOff(div2);
    turnOff(div4);
    turnOff(div5);
}

function clk4(){
    turnOn(div4);
    turnOff(div1);
    turnOff(div2);
    turnOff(div3);
    turnOff(div5);
}

bt.onclick = function(){
    turnOff(div1);
    turnOn(div5);
}

window.onload = function() {
    var image = document.getElementById("img");
    var image2 = document.getElementById("img2");
    var image3 = document.getElementById("img3");
    var image4 = document.getElementById("img4");
    var image5 = document.getElementById("img5");
    var image6 = document.getElementById("img6");
    var image7 = document.getElementById("img7");
    function updateImage() {
       image.src = image.src.split("?")[0] + "?" + new Date().getTime();
       image2.src = image2.src.split("?")[0] + "?" + new Date().getTime();
       image3.src = image3.src.split("?")[0] + "?" + new Date().getTime();
       image4.src = image4.src.split("?")[0] + "?" + new Date().getTime();
       image5.src = image5.src.split("?")[0] + "?" + new Date().getTime();
       image6.src = image6.src.split("?")[0] + "?" + new Date().getTime();
       image7.src = image7.src.split("?")[0] + "?" + new Date().getTime();
    }
    setInterval(updateImage, 2000);
}