fetch('')
.then(response=>{
    if(!response.ok){
        throw new error("Network Error");
    }
    return response.json();
})
.then(data=>console.log(data))
.catch(Error=>Error.log("fetch error"));

async function api() {
try{
    const res = await fetch('apiurl');
    if(!response.ok){
        throw new error("fetch error");
    }
    const data =await response.json();
    console.log(data);
}
catch(error){
    console.error("network error")
}
}

