// index
$(function () {

    // 显示识别结果
    var showResult = function (result) {

        var imageReader = new FileReader();
        imageReader.onload = function (event) {
            $("#resultDiv").empty();

            var showImage = new Image();
            showImage.src = event.target.result;
            showImage.onload = function () {
                var canvas = document.createElement("canvas");
                canvas.width = showImage.width;
                canvas.height = showImage.height;
                var context = canvas.getContext("2d");
                context.strokeStyle = "yellow";
                context.lineWidth = 1;
                context.drawImage(showImage, 0, 0);

                var objDiv = $("<div />");
                result.forEach(function (item) {
                    context.strokeRect(item.x1, item.y1, (item.x2 - item.x1) + 1, (item.y2 - item.y1) + 1);
                    context.strokeText(item.name + "(" + item.probability + "%)", item.x1, item.y1);

                    objDiv.append("<br />")
                    objDiv.append("<span>类别：" + item.id + "，名称："+ item.name +"，可能性：" + item.probability + "%，坐标：(" + item.x1 + ", " + item.y1 + "), (" + item.x2 + ", " + item.y2 + ")。</span>")
                });


                $("#resultDiv").append(canvas);
                $("#resultDiv").append("<br/>");
                $("#resultDiv").append("<span>坐标从图片左上开始，格式为(起始x, 起始y), (结束x, 结束y)，单位为像素。</span>");
                $("#resultDiv").append("<br/>");
                $("#resultDiv").append(objDiv);
            }
        };
        imageReader.readAsDataURL($("#iptImage")[0].files[0])
    };

    // 上传图片
    $("#iptSubmit").click(function () {

        if ($("#iptImage").val() == "") {
            $("#resultDiv").html("<span>请先选择要识别的图片。</span>");
            return false;
        }

        $("#resultDiv").empty();

        $("#uploadForm").find("input").attr("disabled", "disabled")
        var postUrl = $("#uploadForm").attr("action")
        var formData = new FormData();
        formData.append("image", $("#iptImage")[0].files[0])

        $("#resultDiv").html("<span>正在识别图片中物品，识别过程需要一些时间，请稍候。。。</span>");
        $.ajax({
            url: postUrl,
            type: "POST",
            data: formData,
            dataType: 'json',
            processData: false,
            contentType: false})
            .done(showResult)
            .fail(function (req, status) {
                $("#resultDiv").html("<span>识别出错了。。。</span>");
            }).always(function () {
                $("#uploadForm").find("input").removeAttr("disabled");
            });

        return false;
    });

});