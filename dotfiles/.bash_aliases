#... Seldom used, but need to remember
### watch -n1 -x <command>
### free -m
### sudo fuser -v /dev/nvidia*
### fusermount -u /path/to/mount/point

alias unmount='fusermount -u'

alias ll='/bin/ls -altg'
alias lr='/bin/ls -altrg'
alias lss='/bin/ls -altS'
alias j=jobs
alias f=fg
alias m=more
alias cd=sd
alias pe=printenv
alias cx='chmod 755'
alias cw='chmod 744'
alias h=history
alias tf='tail -f'
alias cls='clear'
alias pp2='pp +2'
alias pp3='pp +3'
alias pp4='pp +4'
alias pp5='pp +5'
alias pp6='pp +6'
alias pp7='pp +7'
alias pp8='pp +8'
alias pp9='pp +9'
alias mypath='echo $PATH | tr ":" "\n"'
alias myuc='uc -t -ne -i -b -B -lt'

alias gpumem='sudo fuser -v /dev/nvidia*'

alias python363='source /home/joeantol/python-envs/python363/bin/activate'
alias python354='source /home/joeantol/python-envs/python354/bin/activate'
alias python35gpu='source /home/joeantol/python-envs/python35gpu/bin/activate'
alias projectx='source /home/joeantol/work/project-x/project-x.sh'

alias mypy='rm nohup.out; nohup python -u'

function titleBar
{
   user=`whoami`
   dirs=`dirs`
   hostname=`hostname`

   case $TERM in
       iMac*)
           ;;
       xterm*)
           echo -ne "\033]0;${user}@${hostname}: ${dirs}\007"
           echo -ne "\033]0;${TITLE}: ${dirs}\007"
           ###export PS1="$user@$hostname > "

           ;;
       *)
           TITLEBAR=''
           ;;
   esac

}
function pp {
   pushd $1
   titleBar
}

function po {
   popd
   titleBar
}

function pd {
   titleBar
   dirs | tr " " "\n" | awk '{print NR-1 "  " $0}'
}

function sd {
   \cd $1
   titleBar
}

function pwd {
   echo $PWD
   titleBar
}
