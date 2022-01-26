let send_request = () =>{  
    
    let theUrl = "http://127.0.0.1:8080/";
    var file = document.getElementById("upload").files[0];
    var img_type = document.getElementById("imgType").value;
    theUrl += img_type;
    if(file && (file.type == "image/jpeg" || file.type == "image/jpg" || file.type == "image/png")){
        var req = new XMLHttpRequest();
        var formdata = new FormData();
        formdata.append('media',file); 
        req.onreadystatechange = function() {
            if(req.readyState === 4){
                if(req.status == 200){
                    $('.upload-box').html('<p aria-hidden="true"><div class="custom-upload"><input onchange="readURL(this)" type="file" id="upload"><span>Chosse Image</span></div></p>');
                    // var image = new Image();
                    // image.src = 'data:image/jpg;base64,'+req.responseText;
                    // var w = window.open("","MRB Image",'height=650,width=840');
                    // w.document.write(image.outerHTML);
                    $('#solvedImg').css("display", "inline")
                    $('#solvedImg').attr('src', 'data:image/jpg;base64,'+req.responseText).width("100%").height("100%");
                }
            }
        }
        req.open('POST',theUrl, true);
        req.send(formdata);
    }else{
        alert("Not file uploaded.");
    }
}

function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function(e) {
            $('#uploadedImg').css("display", "inline")
            $('#uploadedImg').attr('src', e.target.result).width("100%").height("100%");
        };
        reader.readAsDataURL(input.files[0]);
    }
    send_request();
}

// var gifs = ['images/2.gif','images/3.gif','images/4.gif','images/1.gif'];
var gifs = ['images/1.gif'];
// var i = 3;
// setInterval(function(){
//     document.getElementById('poster').setAttribute('src',gifs[i])
//     if(i == 3){
//         i = 0;
//     }else{
//         i++;
//     }
// },6000);














    

    
