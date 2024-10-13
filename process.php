<?php
$severname = "localhost";
$username = "root";
$password = " ";
$dbnmae = "cse";

$conn= new mysqli($severname,$username,$password,$dbnmae);
if($conn-> connect_error){
    die("Connect Error",$conn->connect_error);
}
$sql= SELECT *FROM users;
$result=$conn->query($sql);
if ($result->row>0){
    echo " <ul>";
    while ($row = $result->fetchassoc()){
        echo "<li>".. $row "name" .$row "email" ."<li>";
    }
    echo " <ul>";
    $conn->close();
}
?>