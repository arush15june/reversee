console.log( "ready!" );

function sendImage(input) {

	if (input.files && input.files[0]) {
		var reader = new FileReader();

		reader.onload = function (e) {
			console.log(e.target.result);
			$.ajax({
				type: "POST",
				url: 'http://127.0.0.1:5000/l',
				data: e.target.result,
				success: function (res) {
					appendResults(res);
				}
			});
		};
		reader.readAsDataURL(input.files[0]);
	}
};



//Give the path of the image collection folder here
var link_to_folder = '/images';

//argument => complete results object
function appendResults(data) {
	for(i = 0; i < data['results'].length; i++) {
		$('.responser-wrapper').append(
			'<div class="col-md-3 col-6 response-image">'
			+'<img src="'+ link_to_folder + '/' + data['results'][i].file +'">'
			+'<p>'+data['results'][i].loss+'</p>'
			+'</div>'
			);
	}
};