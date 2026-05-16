$(document).ready(function () {
    // Initialisation des Select2
    $('#descriptor').select2();
    $('#image_class').select2();

    // Drag & drop
    $('#drop-area').on('dragover', function (e) {
        e.preventDefault();
        $(this).addClass('bg-light');
    });

    $('#drop-area').on('dragleave drop', function (e) {
        e.preventDefault();
        $(this).removeClass('bg-light');
    });

    $('#drop-area').on('drop', function (e) {
        e.preventDefault();
        const files = e.originalEvent.dataTransfer.files;
        if (files.length) {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(files[0]);
            document.getElementById('file').files = dataTransfer.files;
            uploadFile(files[0]);
        }
    });

    // Clic sur la zone
    $('#drop-area').on('click', function () {
        if (!$('#preview').attr('src')) {
            $('#file').trigger('click');
        }
    });

    // Sélection manuelle
    $('#file').change(function () {
        if (this.files.length) {
            uploadFile(this.files[0]);
        }
    });

    function uploadFile(file) {
        var formData = new FormData();
        formData.append('file', file);

        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (data) {
                if (data.error) {
                    alert(data.error);
                } else {
                    $('#preview').attr('src', data.file_path).removeClass('d-none');
                    $('#filename').val(data.filename);
                    $('#remove-image').removeClass('d-none');
                    $('#drop-zone').addClass('d-none');
                    $('#preview-container').removeClass('d-none');
                    $('button[type="submit"]').prop('disabled', false);

                }
            },
            error: function () {
                alert('Error uploading file.');
            }
        });
    }

    // Fonction centrale de suppression de l'image
    function removeSelectedImage() {
        var filename = $('#filename').val();
        if (!filename) return;

        $.ajax({
            url: '/delete/' + filename,
            type: 'POST',
            success: function (data) {
                if (data.error) {
                    alert(data.error);
                } else {
                    $('#preview').attr('src', '').addClass('d-none');
                    $('#remove-image').addClass('d-none');
                    $('#filename').val('');
                    $('#file').val('');
                    $('#drop-zone').removeClass('d-none');
                    $('#preview-container').addClass('d-none');
                    $('button[type="submit"]').prop('disabled', true);
                    $('#result-container').empty();
                    $('#rp-curve').attr('src', '').addClass('d-none');
                    $('#predicted-container').addClass('d-none');
                    $('#predicted-class').text(''); // Réinitialise la classe affichée
                }
            },
            error: function () {
                alert('Error removing file.');
            }
        });
    }

    // Affichage de l'image en taille réelle lorsque l'utilisateur clique dessus (modale Bootstrap)
    function openModal(imageSrc) {
        $('#modalImage').attr('src', imageSrc);
        $('#imageModal').modal('show');
    }


    // Événement original
    $('#remove-image').click(function (event) {
        event.preventDefault();
        event.stopPropagation();
        removeSelectedImage();
    });

    // Sur changement de descripteur(s), réinitialise les résultats, mais conserve l’image
    $('#descriptor').on('change', function () {
        // Si une image est déjà affichée
        if ($('#filename').val() && $('#preview').attr('src') !== '') {
            // Réinitialiser les résultats seulement
            $('#result-container').empty();
            $('#rp-curve').attr('src', '').addClass('d-none');
            $('#predicted-class').text(''); // Réinitialise la classe affichée
            $('#predicted-container').addClass('d-none');
        }
    });

    // Sur changement de similarity, réinitialise les résultats, mais conserve l’image
    $('#similarity').on('change', function () {
        // Si une image est déjà affichée
        if ($('#filename').val() && $('#preview').attr('src') !== '') {
            // Réinitialiser les résultats seulement
            $('#result-container').empty();
            $('#rp-curve').attr('src', '').addClass('d-none');
            $('#predicted-class').text(''); // Réinitialise la classe affichée
            $('#predicted-container').addClass('d-none');
        }
    });

    // Sur changement de topn, réinitialise les résultats, mais conserve l’image
    $('#topn').on('change', function () {
        // Si une image est déjà affichée
        if ($('#filename').val() && $('#preview').attr('src') !== '') {
            // Réinitialiser les résultats seulement
            $('#result-container').empty();
            $('#rp-curve').attr('src', '').addClass('d-none');
            $('#predicted-class').text(''); // Réinitialise la classe affichée
            $('#predicted-container').addClass('d-none');
        }
    });

    $('#search-form').submit(function (event) {
        event.preventDefault();
        var filename = $('#filename').val();
        if (!filename) {
            alert('Please upload an image first.');
            return;
        }

        $('#result-container').empty();
        $('#rp-curve').attr('src', '').addClass('d-none');
        $('#predicted-class').text(''); // Réinitialise la classe affichée
        $('#loading-spinner').removeClass('d-none');
        $('#search-button').prop('disabled', true);

        $.ajax({
            url: '/search',
            type: 'POST',
            data: $(this).serialize(),
            success: function (data) {
                $('#loading-spinner').addClass('d-none');
                $('#search-button').prop('disabled', false);

                if (data.error) {
                    alert(data.error);
                } else {
                    $('#result-container').empty();
                    
                    let isPlaceholder = data.topn_similar_images.every(img => img === "to_be_completed.jpg");

                    if (isPlaceholder) {
                        data.topn_similar_images.forEach(function (image) {
                            $('#result-container').append('<img src="/static/temp_files/' + image + '" class="result-img">');
                        });
                    } else {
                        data.topn_similar_images.forEach(function (image) {
                            $('#result-container').append('<img src="/static/image.orig/' + image + '" class="result-img">');
                        });
                    }

                    if (data.rp_curve) {
                        var timestamp = new Date().getTime();
                        $('#rp-curve').attr('src', data.rp_curve + "?t=" + timestamp).removeClass('d-none');
                    }

                    // Afficher la classe prédite s'il y en a une
                    const classNames = ["Africa", "Beach", "Buildings", "Buses", "Dinosaurs", "Elephants", "Flowers", "Horses", "Mountains", "Food"];

                    if (data.predicted_class !== undefined && data.predicted_class !== null) {
                        $('#predicted-class').text("Predicted class: " + classNames[data.predicted_class]);
                    } else {
                        $('#predicted-class').text("Predicted class: To be completed");
                    }

                    if (data.specified_class !== undefined && data.specified_class !== null) {
                        $('#specified-class').text("Specified class: " + classNames[data.specified_class]);
                    } else {
                        $('#specified-class').text("Specified class: No class selected");
                    }

                    $('#predicted-container').removeClass('d-none');
                    $('#filename').val(data.filename);
                
                }
            },
            error: function () {
                $('#loading-spinner').addClass('d-none');
                $('#search-button').prop('disabled', false);
                alert('Error performing search.');
            }
        });
    });

});
