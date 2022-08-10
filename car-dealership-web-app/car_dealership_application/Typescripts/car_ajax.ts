// Source: https://stackoverflow.com/a/41023184

export function load() {
    $(document).ready(function () {
        $("#brand-target").on("change", function () {
            let list = $("#model-target");
            $.ajax({
                url: "/Cars/GetModels",
                type: "GET",
                data: { brand: $("#brand-target :selected").text() },
                traditional: true,
                success: function (result) {
                    console.log(result);
                    list.empty();
                    list.append('<option value="">-- Select Model --</option>');
                    $.each(result, function (index, item) {
                        list.append('<option value="' + item['id'] + '"> ' + item['name'] + ' </option>');
                        console.log(item);
                    });
                }
            });
        });
    });
}
