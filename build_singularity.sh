docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
 -v /tmp/test:/work \
    kaczmarj/apptainer build mind-glide_sep2023.sif \
     docker-daemon://armaneshaghi/mind-glide:sep2023 

#cp /tmpt/test/*.sif ./ 
#rm -vf /temp/test/*.sif