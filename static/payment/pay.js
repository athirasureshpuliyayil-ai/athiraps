$(document).ready(function () {
    $('#paymentForm').on('submit', function (e) {
        e.preventDefault(); // Prevent form submission

        // Hide form fields and show animation
        $('.credit-card-form').hide();
        $('#animationContainer').show().css("display", "flex").hide().fadeIn(1000); // Show success message with animation

        // Progress
        var progressBar = $(".loading-bar span");
        var progressAmount = $(".loading-bar").attr("data-progress");
        progressAmount = 0;

        var loadingDelay = setTimeout(function () {
            var interval = setInterval(function () {
                progressAmount += 10;

                progressBar.css("width", progressAmount + "%");

                if (progressAmount >= 100) {
                    setTimeout(function () {
                        clearInterval(interval);
                        reverseAnimation();
                    }, 200);
                }
            }, 200);
        }, 2000);

        // Processing over
        function reverseAnimation() {
            $("#processing").removeClass("uncomplete").addClass("complete");
        }

        // Debug button
        $("#trigger").on("click", function () {
            if ($("#processing.uncomplete").length) {
                $("#processing").removeClass("uncomplete").addClass("complete");
            } else {
                $("#processing").removeClass("complete").addClass("uncomplete");
            }
        });

        setTimeout(() => {
            $('#paymentModal').modal('hide'); // Hide modal
            $('#animationContainer').hide(); // Hide animation container

            // Serialize the form data
            var formData = $('#paymentForm').serialize();
            console.log("Form Data:", formData); // Debug: Log the serialized form data

            // Submit the form using AJAX or manually
            $.ajax({
                url: $('#paymentForm').attr('action'), // Get the form action URL
                method: $('#paymentForm').attr('method'), // Get the form method
                data: formData, // Serialized form data
                success: function (response) {
                    console.log("Form submitted successfully:", response);
                    // Optionally handle response
                },
                error: function (error) {
                    console.error("Error submitting form:", error);
                }
            });

            // Reset form fields if needed
            $('#paymentForm')[0].reset();
        }, 7000); // Adjust time as needed
    });
});