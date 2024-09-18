function onClickedEstimatePrice() {
  console.log("Estimate price button clicked");

  // Get values from form inputs
  var neighborhood = document.getElementById("neighborhood").value;
  var bedrooms = document.getElementById("uiBedrooms").value;
  var beds = document.getElementById("uiBeds").value;
  var accommodates = document.getElementById("uiAccommodates").value;
  var roomType = document.getElementById("room_type").value;
  var minNights = document.getElementById("uiMinNight").value;
  var availability365 = document.getElementById("av365").value;
  var estPrice = document.getElementById("uiEstimatedPrice");

  // URL for POST request
  var url = "http://127.0.0.1:5000/predict_home_price";

  // Send POST request
  $.post(url, {
    neighborhood_group: neighborhood,
    bedrooms: bedrooms,
    beds: beds,
    accommodates: accommodates,
    room_type: roomType,
    minimum_nights: minNights,
    availability_365: availability365
  }, function(data, status) {
    console.log(data.estimated_price);
    estPrice.innerHTML = "<h2> $" + data.estimated_price.toString() + " per night</h2>";
    console.log(status);
  });

  var dashboardBtn = document.getElementById("dashboardBtn");
  dashboardBtn.style.display = "block";
}

function redirectToDashboard() {
  // Open the URL of the dashboard in a new tab
  window.open('http://127.0.0.1:5000/dashboard/', '_blank');
}



